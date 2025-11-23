"""Helper script to run the project's automated test suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest with sensible defaults")
    parser.add_argument("pytest_args", nargs="*", help="Additional arguments forwarded to pytest")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest"]
    if not args.pytest_args:
        cmd.append("-q")
    cmd.extend(args.pytest_args)

    result = subprocess.run(cmd, cwd=ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
