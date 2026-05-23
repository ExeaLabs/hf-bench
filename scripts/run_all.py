#!/usr/bin/env python
"""Run HF-Bench benchmark: all 6 models x 5 seeds.

Usage:
    # Smoke test (fast):
    python scripts/run_all.py --smoke-test

    # Full benchmark:
    python scripts/run_all.py --n-trials 100
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hfbench.constants import MODEL_NAMES, SEEDS
from hfbench.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Run full HF-Bench benchmark.")
    p.add_argument("--smoke-test", action="store_true", help="Quick run: 1 seed, 2 trials")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--data-path", type=Path, default=Path("data/raw/demo_hf.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models to run",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to use",
    )
    return p.parse_args()


def run_command(cmd: list, description: str) -> bool:
    logger.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("FAILED: %s (exit code %d)", description, result.returncode)
        return False
    return True


def main():
    args = parse_args()
    setup_logging()

    # Smoke test overrides
    if args.smoke_test:
        seeds = [args.seeds[0]]
        n_trials = 2
        logger.info("=== SMOKE TEST: 1 seed, 2 Optuna trials per model ===")
    else:
        seeds = args.seeds
        n_trials = args.n_trials
        logger.info(
            "=== FULL BENCHMARK: %d models x %d seeds x %d trials ===",
            len(args.models), len(seeds), n_trials,
        )

    # Ensure demo data exists
    if not args.data_path.exists():
        logger.info("Demo data not found, generating...")
        run_command(
            [sys.executable, str(REPO_ROOT / "scripts" / "make_demo_data.py")],
            "make_demo_data",
        )

    logger.info("NOTE: All outputs are from SYNTHETIC demo data, not real patients.")

    failures = []
    total = len(args.models) * len(seeds)
    done = 0

    for model in args.models:
        for seed in seeds:
            done += 1
            desc = f"{model} seed={seed}"
            logger.info("[%d/%d] Training %s...", done, total, desc)
            success = run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "train_model.py"),
                    "--model", model,
                    "--seed", str(seed),
                    "--n-trials", str(n_trials),
                    "--data-path", str(args.data_path),
                    "--output-dir", str(args.output_dir),
                ],
                desc,
            )
            if not success:
                failures.append(desc)

    # Evaluate all predictions
    logger.info("Evaluating predictions...")
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate.py"),
            "--predictions-dir", str(args.output_dir / "predictions"),
            "--metrics-dir", str(args.output_dir / "metrics"),
        ],
        "evaluate",
    )

    # Aggregate results
    logger.info("Aggregating results...")
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_results.py"),
            "--metrics-dir", str(args.output_dir / "metrics"),
        ],
        "aggregate",
    )

    if failures:
        logger.warning("The following runs FAILED: %s", failures)
    else:
        logger.info("All runs completed successfully.")
    logger.info("Results in: %s [SYNTHETIC DATA ONLY]", args.output_dir)


if __name__ == "__main__":
    main()
