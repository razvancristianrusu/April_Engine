"""Shared machinery for April Engine's closed-loop tests in MuJoCo.

The engine runs the loop. MuJoCo is the plant at the other end of it: it owns the physics
(integration, ground contact, where each actuator force lands), reports the state when the
engine asks, and applies the commands the engine hands back.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from matplotlib.figure import Figure

import april_engine as ae

VEHICLES = Path(__file__).resolve().parent.parent / "vehicles"
GRAVITY = (0.0, 0.0, -9.80665)
TIMESTEP = 1e-3
DURATION = 20.0
VIDEO_FPS = 30
# The stretch at the end of a run over which holding the target is judged.
HOLD_WINDOW = 5.0
FALLEN_TILT = 60.0


@dataclass(frozen=True)
class Conditions:
    """How far the world is from the one the engine was told about. Default: not at all."""

    # Sensors: standard deviation per axis.
    position_noise: float = 0.0  # m
    velocity_noise: float = 0.0  # m/s
    attitude_noise: float = 0.0  # rad
    angular_velocity_noise: float = 0.0  # rad/s
    # Effectors: time constant with which they follow their command.
    effector_lag: float = 0.0  # s
    # Physics: true mass over described mass, and a steady world-frame force on the vehicle.
    mass_scale: float = 1.0
    push: tuple[float, float, float] = (0.0, 0.0, 0.0)  # N


PERFECT = Conditions()
# Sensor noise of the size a decent state estimator leaves behind.
SENSOR_NOISE = {
    "position_noise": 0.02,
    "velocity_noise": 0.05,
    "attitude_noise": 0.01,
    "angular_velocity_noise": 0.02,
}


@dataclass(frozen=True)
class Case:
    """One vehicle, one target. The model's root body must carry its free joint."""

    name: str
    # File in vehicles/ (without .toml) that describes the vehicle to the engine.
    vehicle: str
    model: mujoco.MjModel
    target: tuple[float, float, float]
    # Settled means this close (m) and this slow (m/s).
    tolerance: float
    camera_distance: float
    # Imperfections of a size this vehicle should expect to meet.
    rough: Conditions
    # World axes on which the target is defined; a ground vehicle does not choose its height.
    held_axes: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class Run:
    label: str
    case: Case
    summary: dict[str, object]
    # One row per tick: time, position, tilt, then the command each effector actually got.
    log: np.ndarray
    frames: list[np.ndarray]


def description(vehicle: str) -> str:
    return (VEHICLES / f"{vehicle}.toml").read_text()


def spaced(values) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def scene(name: str, vehicle: str, actuators: str, target: Sequence[float]) -> str:
    """MJCF for a vehicle body on flat ground, with a marker at the target."""
    return f"""
<mujoco model="{name}">
  <option timestep="{TIMESTEP}" gravity="{spaced(GRAVITY)}"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.55 0.6 0.55" rgb2="0.45 0.5 0.45"
             width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 50" dir="0.2 0.2 -1"/>
    <geom name="ground" type="plane" size="0 0 1" material="grid"/>
    <site name="target" pos="{spaced(target)}" size="0.08" rgba="1 0.2 0.2 0.8"/>
    {vehicle}
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>"""


class MujocoPlant:
    """A MuJoCo model as the engine's plant, recording what happens to it."""

    def __init__(self, case: Case, conditions: Conditions, video: bool):
        # The physics may differ from what the engine was told; the case's model must not.
        self.model = copy.copy(case.model)
        self.model.body_mass[:] *= conditions.mass_scale
        self.model.body_inertia[:] *= conditions.mass_scale
        self.data = mujoco.MjData(self.model)
        self.data.xfrc_applied[self.model.jnt_bodyid[0], :3] = conditions.push
        self.log: list[tuple[float, ...]] = []
        self.frames: list[np.ndarray] = []
        self.renderer = mujoco.Renderer(self.model, 720, 1280) if video else None
        self.camera = mujoco.MjvCamera()
        self.camera.distance = case.camera_distance
        self.camera.azimuth, self.camera.elevation = 135, -20

    def measure(self):
        # A free joint reports world-frame linear velocity and body-frame angular velocity,
        # with the quaternion as (w, x, y, z): April Engine's own conventions.
        data = self.data
        return (
            data.qpos[:3].tolist(),
            data.qvel[:3].tolist(),
            data.qpos[3:7].tolist(),
            data.qvel[3:6].tolist(),
        )

    def actuate(self, commands: list[float], period: float) -> None:
        data = self.data
        attitude = data.qpos[3:7]
        upright = np.clip(1 - 2 * (attitude[1] ** 2 + attitude[2] ** 2), -1, 1)
        self.log.append((data.time, *data.qpos[:3], np.degrees(np.arccos(upright)), *commands))
        if self.renderer and data.time * VIDEO_FPS >= len(self.frames):
            self.camera.lookat[:] = data.qpos[:3]
            self.renderer.update_scene(data, self.camera)
            self.frames.append(self.renderer.render())
        data.ctrl[:] = commands
        for _ in range(round(period / self.model.opt.timestep)):
            mujoco.mj_step(self.model, data)


def simulate(
    case: Case,
    conditions: Conditions = PERFECT,
    label: str | None = None,
    video: bool = False,
    seed: int = 0,
) -> Run:
    engine = ae.Engine(description(case.vehicle), case.target)
    plant = MujocoPlant(case, conditions, video)
    engine.run(
        plant,
        round(DURATION / engine.period),
        position_noise=conditions.position_noise,
        velocity_noise=conditions.velocity_noise,
        attitude_noise=conditions.attitude_noise,
        angular_velocity_noise=conditions.angular_velocity_noise,
        effector_lag=conditions.effector_lag,
        seed=seed,
    )

    log = np.array(plant.log)
    axes = list(case.held_axes)
    target = np.array(case.target)[axes]
    path = log[:, 1:4][:, axes]
    holding = log[:, 0] >= DURATION - HOLD_WINDOW
    held = path[holding]
    low, high = case.model.actuator_ctrlrange.T
    at_a_limit = ((log[:, 5:] <= low) | (log[:, 5:] >= high)).any(axis=1)
    miss = float(np.linalg.norm(plant.data.qpos[:3][axes] - target))
    speed = float(np.linalg.norm(plant.data.qvel[:3]))
    label = label or case.name
    summary = {
        "case": label,
        "target_m": case.target,
        "conditions": asdict(conditions),
        "final_miss_m": miss,
        "final_speed_m_s": speed,
        "tolerance": case.tolerance,
        # Where it sits on average at the end, and how much it wanders around that spot.
        "hold_offset_m": float(np.linalg.norm(held.mean(axis=0) - target)),
        "hold_jitter_m": float(np.sqrt(((held - held.mean(axis=0)) ** 2).sum(axis=1).mean())),
        # How hard the effectors work while it sits there: the largest standard deviation of
        # any command, as a fraction of that effector's range.
        "hold_command_swing": float((log[holding, 5:].std(axis=0) / (high - low)).max()),
        "estimated_disturbance_N": list(engine.disturbance),
        "max_tilt_deg": float(log[:, 4].max()),
        "max_overshoot_m": float(max(0.0, (np.sign(target - path[0]) * (path - target)).max())),
        "time_at_a_limit_s": float(at_a_limit.sum() * engine.period),
        "upright": bool(log[:, 4].max() < FALLEN_TILT),
        "settled": miss <= case.tolerance and speed <= case.tolerance,
    }
    return Run(label, case, summary, log, plant.frames)


def save(run: Run, out: Path, data: bool = True) -> None:
    """Writes the run's plot, plus its raw log and video when there are any."""
    plot(run, out / f"{run.label}.png")
    if data:
        header = ["t", "x", "y", "z", "tilt_deg", *(f"u{i}" for i in range(run.log.shape[1] - 5))]
        np.savetxt(
            out / f"{run.label}.csv", run.log, delimiter=",", header=",".join(header), comments=""
        )
    if run.frames:
        import imageio.v2 as imageio

        imageio.mimsave(out / f"{run.label}.mp4", run.frames, fps=VIDEO_FPS)


def plot(run: Run, path: Path) -> None:
    # A bare Figure never touches a GUI backend, so this also works with no display.
    figure = Figure(figsize=(9, 7))
    top, bottom = figure.subplots(2, 1, sharex=True)
    log, case = run.log, run.case
    for axis, label in enumerate("xyz"):
        (line,) = top.plot(log[:, 0], log[:, 1 + axis], label=label)
        if axis in case.held_axes:
            top.axhline(case.target[axis], color=line.get_color(), linestyle=":", linewidth=1)
    top.set_ylabel("position (m), targets dotted")
    top.set_title(f"{run.label}: April Engine in MuJoCo")
    top.legend()
    for index in range(log.shape[1] - 5):
        bottom.plot(log[:, 0], log[:, 5 + index], label=f"u{index}")
    bottom.set_ylabel("effector command (N)")
    bottom.set_xlabel("time (s)")
    bottom.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
