"""Does April Engine still hold its target when the world is not as described?

Every vehicle is flown once in a perfect world, once under each imperfection alone, and once
under all of them together. A run passes if the vehicle stays upright, ends up sitting on its
target, sits still, and does not thrash its effectors to do so.

    python robustness.py --out results/robustness

Exits non-zero if any run fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from flyers import quadrotor, rocket
from harness import PERFECT, Case, Conditions, save, simulate
from rover import rover_forward

CASES = (quadrotor, rocket, rover_forward)
# Judged over the last seconds of each run.
MAX_HOLD_OFFSET = 0.05  # m, average distance from the target
MAX_HOLD_JITTER = 0.05  # m, wander around that average
MAX_COMMAND_SWING = 0.15  # fraction of an effector's range


def conditions(rough: Conditions) -> dict[str, Conditions]:
    return {
        "perfect": PERFECT,
        "noise": replace(
            PERFECT,
            position_noise=rough.position_noise,
            velocity_noise=rough.velocity_noise,
            attitude_noise=rough.attitude_noise,
            angular_velocity_noise=rough.angular_velocity_noise,
        ),
        "lag": replace(PERFECT, effector_lag=rough.effector_lag),
        "heavy": replace(PERFECT, mass_scale=rough.mass_scale),
        "push": replace(PERFECT, push=rough.push),
        "all": rough,
    }


def passes(summary: dict) -> bool:
    return (
        summary["upright"]
        and summary["hold_offset_m"] <= MAX_HOLD_OFFSET
        and summary["hold_jitter_m"] <= MAX_HOLD_JITTER
        and summary["hold_command_swing"] <= MAX_COMMAND_SWING
    )


def table(summaries: list[dict]) -> str:
    rows = [
        "| run | hold offset m | hold jitter m | command swing % | max tilt ° | at a limit s | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        rows.append(
            f"| {summary['case']} | {summary['hold_offset_m']:.3f} | {summary['hold_jitter_m']:.3f} "
            f"| {100 * summary['hold_command_swing']:.1f} | {summary['max_tilt_deg']:.1f} "
            f"| {summary['time_at_a_limit_s']:.2f} | {'ok' if summary['passes'] else 'FAIL'} |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="April Engine robustness tests in MuJoCo")
    parser.add_argument("--out", type=Path, default=Path("results/robustness"))
    parser.add_argument("--video", action="store_true", help="also render an mp4 per run")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for build in CASES:
        case: Case = build()
        for name, condition in conditions(case.rough).items():
            run = simulate(case, condition, f"{case.name}_{name}", arguments.video)
            save(run, arguments.out, data=False)
            summaries.append({**run.summary, "passes": passes(run.summary)})

    (arguments.out / "robustness.json").write_text(json.dumps(summaries, indent=2))
    (arguments.out / "robustness.md").write_text(table(summaries) + "\n")
    print(table(summaries))
    return 0 if all(summary["passes"] for summary in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
