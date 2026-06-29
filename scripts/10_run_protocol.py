from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n" + "=" * 100)
    print("Running:", " ".join(command))
    print("=" * 100)

    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full prepared evaluation protocol.")
    parser.add_argument("--overwrite-stress", action="store_true")
    parser.add_argument("--skip-size-eval", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    python = sys.executable

    stress_cmd = [
        python,
        "scripts/08_make_stress_tests.py",
    ]

    if args.overwrite_stress:
        stress_cmd.append("--overwrite")

    run(stress_cmd)

    eval_cmd = [
        python,
        "scripts/09_evaluate_protocol.py",
        "--splits",
        *args.splits,
    ]

    if args.skip_size_eval:
        eval_cmd.append("--skip-size-eval")

    run(eval_cmd)

    print("\nFull protocol completed successfully.")


if __name__ == "__main__":
    main()
