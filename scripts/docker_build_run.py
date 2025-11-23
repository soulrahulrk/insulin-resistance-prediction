"""Build (and optionally run) the Insulin Resistance API Docker image."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print(f"[docker] Executing: {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and optionally run the Docker image")
    parser.add_argument("--tag", default="ir-api:latest", help="Docker image tag")
    parser.add_argument("--no-cache", action="store_true", help="Disable Docker build cache")
    parser.add_argument("--run", action="store_true", help="Run the container after building")
    parser.add_argument("--port", default="8000", help="Host port to expose when running the container")
    parser.add_argument(
        "--env", action="append", default=[], help="Environment variables to pass when running (KEY=VALUE)"
    )
    args = parser.parse_args()

    build_cmd = ["docker", "build", "-t", args.tag]
    if args.no_cache:
        build_cmd.append("--no-cache")
    build_cmd.append(str(ROOT))
    run_command(build_cmd)

    if args.run:
        run_cmd = [
            "docker",
            "run",
            "--rm",
            "-p",
            f"{args.port}:8000",
        ]
        for env_item in args.env:
            run_cmd.extend(["-e", env_item])
        run_cmd.extend([args.tag])
        run_command(run_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
