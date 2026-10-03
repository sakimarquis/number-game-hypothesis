"""Durable per-request progress used by checkpointed online text-trial scoring."""

from __future__ import annotations

import os
import shutil

from utils import load_json, save_json


PROGRESS_KIND = "posterior_text_trials_progress"
INITIAL_PHASE = "initial"
RECOVERY_PHASES = ("recovery-1", "recovery-2")
PHASES = (INITIAL_PHASE, *RECOVERY_PHASES)


def _progress_conflict(path: str, reason: str) -> RuntimeError:
    return RuntimeError(
        f"Text-posterior progress is {reason} and was left unchanged: {path}. "
        "Use --refresh-raw-cache to deliberately discard incompatible progress."
    )


def remove_text_trial_progress(path: str, *, expected_identity: dict | None = None) -> bool:
    """Remove one exact progress directory, optionally only for a matching identity."""
    if not os.path.isdir(path):
        return False
    if expected_identity is not None:
        manifest = load_json(os.path.join(path, "manifest.json"))
        if manifest is None or manifest.get("cache_identity") != expected_identity:
            return False
    shutil.rmtree(path)
    return True


class TextTrialProgress:
    """One condition's (phase, prefix, trial) response texts, optionally mirrored to disk.

    The manifest binds progress to the exact raw-cache identity, prefixes, trial
    count, domain, and text format; any mismatch fails closed.
    """

    def __init__(
        self,
        path: str | None,
        *,
        cache_identity: dict | None,
        text_format: str,
        prefixes: list[list[int]],
        n_trials: int,
        domain: int,
    ):
        if path is not None and cache_identity is None:
            raise ValueError("Durable text-trial progress requires a cache identity")
        self.path = path
        self.prefixes = [list(prefix) for prefix in prefixes]
        self._manifest = {
            "kind": PROGRESS_KIND,
            "cache_identity": cache_identity,
            "text_format": text_format,
            "prefixes": self.prefixes,
            "n_trials": n_trials,
            "domain": domain,
        }
        self._plans: dict[str, list[int]] = {}
        self._texts: dict[tuple[str, int, int], str] = {}
        if path is not None:
            self._load_or_create()

    def _load_or_create(self) -> None:
        manifest_path = os.path.join(self.path, "manifest.json")
        if not os.path.exists(self.path):
            save_json(manifest_path, self._manifest)
            return
        if load_json(manifest_path) != self._manifest:
            raise _progress_conflict(self.path, "incompatible with the requested semantics")

        for phase in RECOVERY_PHASES:
            plan = load_json(self._plan_path(phase))
            if plan is not None:
                self._plans[phase] = list(plan["prefix_indices"])
        for phase in PHASES:
            phase_dir = os.path.join(self.path, phase)
            if not os.path.isdir(phase_dir):
                continue
            for name in os.listdir(phase_dir):
                record = load_json(os.path.join(phase_dir, name))
                if record["prefix"] != self.prefixes[record["prefix_index"]]:
                    raise _progress_conflict(self.path, f"containing mismatched prefix record {name}")
                self._texts[(phase, record["prefix_index"], record["trial_index"])] = record["text"]

    def _plan_path(self, phase: str) -> str:
        return os.path.join(self.path, f"{phase}_plan.json")

    def _result_path(self, phase: str, prefix_index: int, trial_index: int) -> str:
        return os.path.join(self.path, phase, f"p{prefix_index:06d}_t{trial_index:04d}.json")

    def _save_plan(self, phase: str, plan: list[int]) -> None:
        if self.path is not None:
            save_json(self._plan_path(phase), {"phase": phase, "prefix_indices": plan})

    def ensure_recovery_plan(self, phase: str, prefix_indices: list[int]) -> list[int]:
        """Extend the persisted plan for a recovery phase with newly missing prefixes."""
        plan = list(self._plans.get(phase, []))
        plan.extend(index for index in prefix_indices if index not in plan)
        if plan != self._plans.get(phase):
            self._save_plan(phase, plan)
            self._plans[phase] = plan
        return list(plan)

    def has_result(self, phase: str, prefix_index: int, trial_index: int) -> bool:
        return (phase, prefix_index, trial_index) in self._texts

    def result_text(self, phase: str, prefix_index: int, trial_index: int) -> str:
        return self._texts[(phase, prefix_index, trial_index)]

    def record_result(self, phase: str, prefix_index: int, trial_index: int, text: str) -> None:
        key = (phase, prefix_index, trial_index)
        existing = self._texts.get(key)
        if existing is not None:
            if existing != text:
                raise _progress_conflict(self.path or "<memory>", f"conflicting result for {key}")
            return
        if self.path is not None:
            save_json(self._result_path(phase, prefix_index, trial_index), {
                "phase": phase,
                "prefix_index": prefix_index,
                "trial_index": trial_index,
                "prefix": self.prefixes[prefix_index],
                "text": text,
            })
        self._texts[key] = text

    def reset_prefixes(self, prefix_indices: list[int]) -> None:
        """Archive and reset exhausted prefixes so a later process can try them from scratch."""
        reset = set(prefix_indices)
        if self.path is not None:
            archive_path = os.path.join(self.path, "exhausted_attempts.json")
            archive = load_json(archive_path) or {"attempts": []}
            archive["attempts"].extend(
                {
                    "prefix_index": prefix_index,
                    "prefix": self.prefixes[prefix_index],
                    "responses": [
                        {"phase": phase, "trial_index": trial_index, "text": text}
                        for (phase, record_prefix, trial_index), text in self._texts.items()
                        if record_prefix == prefix_index
                    ],
                }
                for prefix_index in sorted(reset)
            )
            save_json(archive_path, archive)
        for key in [key for key in self._texts if key[1] in reset]:
            if self.path is not None:
                os.remove(self._result_path(*key))
            del self._texts[key]

        for phase, plan in list(self._plans.items()):
            retained = [prefix_index for prefix_index in plan if prefix_index not in reset]
            if retained:
                self._save_plan(phase, retained)
                self._plans[phase] = retained
            else:
                if self.path is not None:
                    os.remove(self._plan_path(phase))
                del self._plans[phase]
