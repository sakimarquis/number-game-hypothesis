"""Shared CLI argument helpers for number-game entrypoints."""

from __future__ import annotations

import argparse

from configs.backend_registry import BACKEND_CHOICES
from configs.experiment import load_experiment


def add_common_cli_args(parser: argparse.ArgumentParser, *, model_required: bool) -> None:
    parser.add_argument("--model", "-m", required=model_required, default=None, help="Model alias or display name")
    parser.add_argument("--backend", "-b", default=None, choices=BACKEND_CHOICES,
                        help="Backend override; defaults to the model alias's default_backend.")
    parser.add_argument("--thinking", action="store_true", default=False,
                        help="Enable or analyze thinking-mode results for backends that support it")


def experiment_from_args(args: argparse.Namespace, *, run_profile: str | None = None):
    return load_experiment(
        backend=args.backend, thinking=args.thinking,
        model_override=args.model, run_profile=run_profile,
    )
