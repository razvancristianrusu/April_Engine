"""Thrust-vectored vehicles: a quadrotor and a rocket on a gimbaled engine.

Both start resting on the ground and must settle on a target in the air. Their MuJoCo models
are generated from the same description files the engine reads, so the two cannot disagree
about where an effector sits; what MuJoCo does with a force once applied is its own business.
"""

from __future__ import annotations

import tomllib

import mujoco

from harness import SENSOR_NOISE, Case, Conditions, description, scene, spaced


def actuator(index: int, effector: dict) -> tuple[str, str]:
    """A site on the body and a motor acting through it, for one described effector."""
    if effector["kind"] == "rotor":
        direction, low = effector["axis"], 0.0
        reaction = [effector["torque_per_thrust"] * component for component in direction]
    else:
        direction, low = effector["direction"], effector["min"]
        reaction = [0.0, 0.0, 0.0]
    site = f'<site name="e{index}" pos="{spaced(effector["position"])}"/>'
    motor = (
        f'<motor site="e{index}" gear="{spaced(direction)} {spaced(reaction)}" '
        f'ctrllimited="true" ctrlrange="{low} {effector["max"]}"/>'
    )
    return site, motor


def flyer(
    name: str,
    hull: str,
    rest_height: float,
    target: tuple[float, float, float],
    camera_distance: float,
    rough: Conditions,
) -> Case:
    described = tomllib.loads(description(name))
    body = described["body"]
    sites, motors = zip(*(actuator(i, e) for i, e in enumerate(described["effector"])))
    vehicle = f"""
    <body name="vehicle" pos="0 0 {rest_height}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{body["mass"]}"
                diaginertia="{spaced(body["inertia"][i][i] for i in range(3))}"/>
      {hull}
      {"".join(sites)}
    </body>"""
    return Case(
        name=name,
        vehicle=name,
        model=mujoco.MjModel.from_xml_string(scene(name, vehicle, "".join(motors), target)),
        target=target,
        tolerance=0.02,
        camera_distance=camera_distance,
        rough=rough,
    )


def quadrotor() -> Case:
    return flyer(
        name="quadrotor",
        hull='<geom type="box" size="0.18 0.18 0.03" rgba="0.2 0.45 0.8 1"/>',
        rest_height=0.03,
        target=(4.0, -2.0, 3.0),
        camera_distance=3.0,
        # Motors that take 50 ms to spin up, 20 % of payload, a 2 N side wind.
        rough=Conditions(**SENSOR_NOISE, effector_lag=0.05, mass_scale=1.2, push=(2.0, 0.0, 0.0)),
    )


def rocket() -> Case:
    return flyer(
        name="rocket",
        hull='<geom type="cylinder" fromto="0 0 -1 0 0 1" size="0.2" rgba="0.85 0.85 0.9 1"/>',
        rest_height=1.0,
        target=(6.0, 3.0, 20.0),
        camera_distance=9.0,
        # A gimbal and throttle that take 100 ms, 20 % more propellant, a 30 N side wind.
        rough=Conditions(**SENSOR_NOISE, effector_lag=0.1, mass_scale=1.2, push=(30.0, 0.0, 0.0)),
    )
