"""April Engine driving every vehicle it knows inside MuJoCo.

    python mujoco_tests.py --out results [--video] [--only rocket rover_forward]

Exits non-zero if a vehicle does not settle on its target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flyers import quadrotor, rocket
from harness import save, simulate
from rover import rover_forward, rover_reverse

CASES = {case.__name__: case for case in (quadrotor, rocket, rover_forward, rover_reverse)}


def main() -> int:
    parser = argparse.ArgumentParser(description="April Engine closed-loop tests in MuJoCo")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--video", action="store_true", help="also render an mp4 per case")
    parser.add_argument("--only", nargs="+", choices=CASES, default=list(CASES), metavar="CASE")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in arguments.only:
        run = simulate(CASES[name](), video=arguments.video)
        save(run, arguments.out)
        summaries.append(run.summary)
    (arguments.out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    return 0 if all(summary["settled"] for summary in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
