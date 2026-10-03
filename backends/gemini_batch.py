"""Resumable Gemini Developer API Batch lifecycle for GenerateContent requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Callable

from google.genai import types

from utils import load_json, save_json


_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
_MAX_PROVIDER_RETRIES = 2


def _state_name(state) -> str:
    name = getattr(state, "name", None)
    return name or str(state).rsplit(".", 1)[-1]


def _file_request(request) -> dict:
    raw = request.model_dump(mode="json", by_alias=False, exclude_none=True)
    key = raw.pop("metadata")["key"]
    config = raw.pop("config", {})
    system_instruction = config.pop("system_instruction", None)
    if system_instruction is not None:
        if isinstance(system_instruction, str):
            system_instruction = {"parts": [{"text": system_instruction}]}
        raw["system_instruction"] = system_instruction
    if config:
        raw["generation_config"] = config
    return {"key": key, "request": raw}


class GeminiBatchExecutor:
    """Submit, resume, collect, and compact one Gemini Batch job per request set.

    Each run directory is keyed by batch key and request hash. Its manifest records
    the uploaded file and the non-idempotent job name before polling, so an
    interrupted rerun resumes the job instead of submitting it again.
    """

    def __init__(self, client, model_id: str, root: str, *, priority: int = 0,
                 poll_interval_seconds: float = 30.0):
        self.client = client
        self.model_id = model_id
        self.root = Path(root).resolve()
        self.priority = priority
        self.poll_interval_seconds = poll_interval_seconds

    def _stage_requests(self, requests: list, batch_key: str) -> tuple[Path, Path, str]:
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".staging-{uuid.uuid4().hex}.jsonl"
        digest = hashlib.sha256()
        with staging.open("wb") as handle:
            for request in requests:
                row = json.dumps(
                    _file_request(request),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8") + b"\n"
                digest.update(row)
                handle.write(row)

        request_hash = digest.hexdigest()
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", batch_key).strip("-")[:64] or "batch"
        return staging, self.root / f"{slug}-{request_hash[:12]}", request_hash

    async def _poll(self, job, manifest: dict, manifest_path: Path):
        while _state_name(job.state) not in _TERMINAL_STATES:
            state = _state_name(job.state)
            if manifest.get("state") != state:
                print(f"    [gemini batch] {job.name}: {state}")
                manifest["state"] = state
                save_json(str(manifest_path), manifest)
            await asyncio.sleep(self.poll_interval_seconds)
            job = await self.client.aio.batches.get(name=job.name)

        state = _state_name(job.state)
        manifest["state"] = state
        save_json(str(manifest_path), manifest)
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini Batch {job.name} ended in {state}")
        return job

    async def _delete_remote_files(self, manifest: dict, manifest_path: Path) -> None:
        """Delete uploaded input/output files, keeping names whose deletion failed."""
        for field in ("input_file", "output_file"):
            name = manifest.get(field)
            if not name:
                continue
            try:
                await self.client.aio.files.delete(name=name)
                manifest[field] = None
            except Exception as exc:
                print(f"    [gemini batch] could not delete {name}: {exc}")
        save_json(str(manifest_path), manifest)

    def _decode_output(self, output: bytes, expected: int, decode: Callable) -> tuple[list, list[int]]:
        decoded: dict[str, object] = {}
        diagnostics = []
        for raw_line in output.splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            key = str(row.get("key", ""))
            if row.get("error") is not None or row.get("response") is None:
                continue
            try:
                response = types.GenerateContentResponse.model_validate(row["response"], extra="ignore")
                decoded[key] = decode(response)
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                diagnostics.append(f"{key}: {exc}")

        if diagnostics:
            print(
                f"    [gemini batch] {len(diagnostics)} rows failed response decoding; "
                f"first errors: {diagnostics[:3]}"
            )
        failed = [index for index in range(expected) if str(index) not in decoded]
        return [decoded.get(str(index)) for index in range(expected)], failed

    async def _find_submitted_job(self, display_name: str):
        pager = await self.client.aio.batches.list(config={"page_size": 1000})
        matches = [
            job
            async for job in pager
            if getattr(job, "display_name", None) == display_name
        ]
        if len(matches) > 1:
            names = [job.name for job in matches]
            raise RuntimeError(
                f"Multiple Gemini Batch jobs use display_name {display_name!r}: {names}"
            )
        return matches[0] if matches else None

    async def _submit(self, run_dir: Path, request_path: Path, manifest: dict, manifest_path: Path, n_requests: int):
        if manifest.get("input_file") is None:
            uploaded = await self.client.aio.files.upload(
                file=request_path,
                config=types.UploadFileConfig(display_name=run_dir.name, mime_type="jsonl"),
            )
            manifest["input_file"] = uploaded.name
            manifest["state"] = "UPLOADED"
            save_json(str(manifest_path), manifest)

        if manifest.get("state") == "SUBMITTING":
            job = await self._find_submitted_job(run_dir.name)
            if job is None:
                raise RuntimeError(
                    f"Gemini Batch submission state is ambiguous for {run_dir.name}. "
                    "No new job was submitted; rerun after the remote job becomes visible "
                    "or inspect the manifest and Gemini batch list."
                )
        else:
            manifest["state"] = "SUBMITTING"
            save_json(str(manifest_path), manifest)
            create_config = {"display_name": run_dir.name}
            if self.priority:
                # google-genai has no typed Batch priority field; extra_body merges
                # into the generated Batch resource before submission.
                create_config["http_options"] = types.HttpOptions(
                    extra_body={"batch": {"priority": str(self.priority)}}
                )
            job = await self.client.aio.batches.create(
                model=self.model_id,
                src=manifest["input_file"],
                config=types.CreateBatchJobConfig(**create_config),
            )
        manifest["job_name"] = job.name
        manifest["state"] = _state_name(job.state)
        save_json(str(manifest_path), manifest)
        print(f"    [gemini batch] created {job.name} ({n_requests} requests)")
        return job

    async def run(
        self,
        requests: list,
        *,
        batch_key: str,
        decode: Callable,
        provider_attempt: int = 0,
    ) -> list:
        if not requests:
            return []

        staging, run_dir, request_hash = await asyncio.to_thread(self._stage_requests, requests, batch_key)
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / "requests.jsonl"
        response_path = run_dir / "responses.jsonl"
        normalized_path = run_dir / "normalized.json"
        manifest_path = run_dir / "manifest.json"
        manifest = load_json(str(manifest_path)) or {}

        if normalized_path.exists():
            staging.unlink()
            values = load_json(str(normalized_path))
            if len(values) != len(requests):
                raise RuntimeError(f"Invalid compacted Gemini Batch output: {normalized_path}")
            return values

        if manifest and manifest.get("request_hash") != request_hash:
            staging.unlink()
            raise RuntimeError(f"Gemini Batch manifest hash mismatch: {manifest_path}")
        if request_path.exists():
            staging.unlink()
        else:
            os.replace(staging, request_path)

        if not manifest:
            manifest = {
                "schema": 1,
                "model": self.model_id,
                "batch_key": batch_key,
                "request_hash": request_hash,
                "request_count": len(requests),
                "state": "LOCAL_PREPARED",
            }
            save_json(str(manifest_path), manifest)

        job_name = manifest.get("job_name")
        if job_name:
            job = await self.client.aio.batches.get(name=job_name)
        else:
            job = await self._submit(run_dir, request_path, manifest, manifest_path, len(requests))

        job = await self._poll(job, manifest, manifest_path)
        output_file = getattr(getattr(job, "dest", None), "file_name", None)
        manifest["output_file"] = output_file
        save_json(str(manifest_path), manifest)
        if response_path.exists():
            output = await asyncio.to_thread(response_path.read_bytes)
        else:
            if not output_file:
                raise RuntimeError(f"Gemini Batch {job.name} succeeded without an output file")
            output = await self.client.aio.files.download(file=output_file)
            temporary = response_path.with_name(f".{response_path.name}.tmp")
            await asyncio.to_thread(temporary.write_bytes, output)
            os.replace(temporary, response_path)

        values, failed = await asyncio.to_thread(self._decode_output, output, len(requests), decode)
        if failed:
            if provider_attempt >= _MAX_PROVIDER_RETRIES:
                raise RuntimeError(
                    f"Gemini Batch still has {len(failed)} failed rows after "
                    f"{provider_attempt} provider retries"
                )
            print(
                f"    [gemini batch] retrying {len(failed)} failed rows "
                f"(attempt {provider_attempt + 1}/{_MAX_PROVIDER_RETRIES})"
            )
            retry_requests = []
            for retry_index, original_index in enumerate(failed):
                request = requests[original_index].model_copy(deep=True)
                request.metadata = {"key": str(retry_index)}
                retry_requests.append(request)
            retried = await self.run(
                retry_requests,
                batch_key=f"{batch_key}-provider-retry-{provider_attempt + 1}",
                decode=decode,
                provider_attempt=provider_attempt + 1,
            )
            for original_index, value in zip(failed, retried):
                values[original_index] = value

        # Compact while the condition is still pending; the run directory is removed
        # by cleanup() once the canonical cache is published.
        save_json(str(normalized_path), values)
        await self._delete_remote_files(manifest, manifest_path)
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
        return values

    async def cleanup(self, batch_key_prefix: str) -> None:
        """Remove completed run directories for one published condition."""
        if not self.root.exists():
            return
        for manifest_path in self.root.glob("*/manifest.json"):
            run_dir = manifest_path.parent
            manifest = load_json(str(manifest_path))
            if not str(manifest.get("batch_key", "")).startswith(batch_key_prefix):
                continue
            if not (run_dir / "normalized.json").exists():
                continue
            await self._delete_remote_files(manifest, manifest_path)
            if manifest.get("input_file") or manifest.get("output_file"):
                print(f"    [gemini batch] preserving cleanup-pending sidecar {run_dir}")
                continue
            for name in ("normalized.json", "manifest.json", "requests.jsonl", "responses.jsonl"):
                (run_dir / name).unlink(missing_ok=True)
            run_dir.rmdir()
