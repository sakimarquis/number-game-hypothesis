"""Cross-condition analysis runner for number game experiments.

Loads all result JSONs for a model, computes divergence / alpha_beta /
prompt-effects / trajectory metrics, and generates aggregate plots.

Usage:
    python run_analysis.py --model gpt54-mini
    python run_analysis.py --model bayesian
"""

import argparse

from configs.cli import add_common_cli_args, experiment_from_args
from configs.settings import is_baseline_run
from analysis import bayesian
from analysis.pipeline import run_all_stages


def main():
    parser = argparse.ArgumentParser(description="Cross-condition analysis of number game results")
    add_common_cli_args(parser, model_required=True)
    parser.add_argument("--include-per-condition", action="store_true", default=False,
                        help="Also rerun the expensive per-condition plots/metrics before aggregate stages")
    args = parser.parse_args()

    experiment = experiment_from_args(args)
    if is_baseline_run(args.model):
        bayesian.run_baseline_analyses(experiment)
        print("\nAnalysis complete.")
        return

    run_all_stages(experiment, include_per_condition=args.include_per_condition)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
