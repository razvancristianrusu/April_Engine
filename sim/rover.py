"""A differential-drive rover: two driven wheels, two frictionless casters.

Unlike the flyers, nothing here is applied to the body directly. The engine asks each wheel
for a traction force; MuJoCo turns that into a torque on the wheel's axle, and only friction
with the ground turns it into motion. Wheel inertia and slip are not in the engine's model.

The model below is built by hand, and vehicles/rover.toml describes the same machine to the
engine as one rigid body. `rover` refuses to run if the two have drifted apart.
"""

from __future__ import annotations

import tomllib

import mujoco

from harness import SENSOR_NOISE, Case, Conditions, description, scene

WHEEL_RADIUS = 0.08
HALF_TRACK = 0.18
AXLE_HEIGHT = -0.02  # wheel axle relative to the chassis origin


def wheel(name: str, y: float, max_traction: float) -> tuple[str, str]:
    body = f"""
      <body name="{name}" pos="0 {y} {AXLE_HEIGHT}">
        <joint name="{name}" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" fromto="0 -0.02 0 0 0.02 0" size="{WHEEL_RADIUS}" mass="0.5"
              rgba="0.15 0.15 0.15 1"/>
      </body>"""
    # gear = wheel radius, so the control signal is traction force in newtons.
    motor = (
        f'<motor joint="{name}" gear="{WHEEL_RADIUS}" ctrllimited="true" '
        f'ctrlrange="{-max_traction} {max_traction}"/>'
    )
    return body, motor


def caster(x: float) -> str:
    # condim 1 with priority over the ground makes the contact frictionless: it only carries
    # weight, like a ball caster.
    return (
        f'<geom type="sphere" pos="{x} 0 -0.07" size="0.03" mass="0.1" condim="1" priority="1" '
        f'rgba="0.6 0.6 0.6 1"/>'
    )


def rover(name: str, target: tuple[float, float]) -> Case:
    described = tomllib.loads(description("rover"))
    left_wheel, right_wheel = described["effector"]
    left, left_motor = wheel("left", HALF_TRACK, left_wheel["max"])
    right, right_motor = wheel("right", -HALF_TRACK, right_wheel["max"])
    vehicle = f"""
    <body name="vehicle" pos="0 0 {WHEEL_RADIUS - AXLE_HEIGHT}">
      <freejoint/>
      <geom type="box" size="0.2 0.15 0.05" mass="10" rgba="0.85 0.55 0.15 1"/>
      <geom type="box" pos="0.16 0 0.055" size="0.04 0.04 0.005" mass="0" contype="0"
            conaffinity="0" rgba="0.9 0.1 0.1 1"/>
      {caster(0.17)}
      {caster(-0.17)}
      {left}
      {right}
    </body>"""
    goal = (*target, 0.0)
    model = mujoco.MjModel.from_xml_string(scene(name, vehicle, left_motor + right_motor, goal))

    mass, track = float(model.body_mass.sum()), left_wheel["position"][1]
    if abs(mass - described["body"]["mass"]) > 1e-6 or abs(track - HALF_TRACK) > 1e-6:
        raise ValueError(
            f"rover.toml (mass {described['body']['mass']}, half track {track}) no longer "
            f"describes this model (mass {mass}, half track {HALF_TRACK})"
        )
    return Case(
        name=name,
        vehicle="rover",
        model=model,
        target=goal,
        tolerance=0.05,
        camera_distance=2.5,
        # Drives that take 30 ms, 20 % of cargo, a steady 3 N shove along and across.
        rough=Conditions(**SENSOR_NOISE, effector_lag=0.03, mass_scale=1.2, push=(3.0, 3.0, 0.0)),
        held_axes=(0, 1),
    )


def rover_forward() -> Case:
    return rover("rover_forward", (3.0, 2.0))


def rover_reverse() -> Case:
    return rover("rover_reverse", (-2.0, -1.0))
