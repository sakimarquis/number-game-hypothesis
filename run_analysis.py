"""Refresh default condition and aggregate analysis stages for one LLM from its raw caches."""

import argparse

from configs.cli import add_common_cli_args, experiment_from_args
from configs.settings import is_baseline_run
from analysis.pipeline import run_all_stages


def main():
    parser = argparse.ArgumentParser(description="Cross-condition analysis of number game results")
    add_common_cli_args(parser, model_required=True)
    parser.add_argument("--include-per-condition", action="store_true", default=False,
                        help="Also rerun the expensive per-condition plots/metrics before aggregate stages")
    args = parser.parse_args()
    if is_baseline_run(args.model):
        parser.error("baselines are built by `run_number_game.py --model bayesian`")

    run_all_stages(experiment_from_args(args), include_per_condition=args.include_per_condition)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
