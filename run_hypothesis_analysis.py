"""Structure generations, then refresh condition-local and aggregate hypothesis analysis.

Usage:
    python run_hypothesis_analysis.py --model gpt54-mini
"""

import argparse
from pathlib import Path

from configs.cli import add_common_cli_args, experiment_from_args
from configs.layout import model_root
from analysis.hypothesis_structuring import structure_results
from analysis.pipeline import run_hypothesis_stages
from analysis.summary import run_hypothesis_summary


def main():
    parser = argparse.ArgumentParser(description="Hypothesis analysis of number game results")
    add_common_cli_args(parser, model_required=True)
    args = parser.parse_args()

    experiment = experiment_from_args(args)
    model_dir = model_root(experiment.backend.result_key)
    stats = structure_results(Path(model_dir))
    for domain, counts in sorted(stats.items()):
        print(f"Structured d{domain}: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    run_hypothesis_stages(experiment)
    run_hypothesis_summary()

    print("\nHypothesis analysis complete.")


if __name__ == "__main__":
    main()
