<p align="center">
  <img src="assets/banner.png" alt="April Engine" width="800">
</p>

# April Engine

One control engine for anything that moves: multirotors, rockets, ground rovers, and
eventually legged machines.

Every vehicle is reduced to the same three parts:

1. a **rigid body** with a state (position, velocity, attitude, angular velocity);
2. a set of **effectors**, each turning one scalar command into a force and torque on that
   body: a rotor, one component of a gimbaled thrust, a wheel's traction;
3. a **control law** that asks for a force and torque on the body.

An allocator finds the effector commands that best deliver what the law asks for, inside
each effector's limits. A new kind of vehicle is a new effector layout and, at most, a new
control law. The kernel underneath does not change.

## Status

Early, and simulation only. Nothing here has driven real hardware yet.

| Vehicle | Control law | Verified in |
|---|---|---|
| Quadrotor | thrust vectoring | built-in rigid-body plant, MuJoCo |
| Rocket on a gimbaled engine | thrust vectoring | built-in rigid-body plant, MuJoCo |
| Differential-drive rover | ground drive | MuJoCo |

The only behaviour so far is "go to a point and hold it". Not there yet: state estimation,
safety (arming, emergency stop, failsafes), trajectories, legged robots, cars with steered
wheels, torque disturbances.

## How it is put together

```
vehicles/*.toml ──► Engine ◄──── measure ──── Plant (simulator or real machine)
                      │                          ▲
                      └──────── actuate ─────────┘
```

- **A vehicle is data.** `vehicles/quadrotor.toml`, `rocket.toml` and `rover.toml` hold the
  mass, inertia, effectors, allocation priorities, control law and gains.
- **The engine owns the loop.** Each tick it measures, decides and actuates through a
  `Plant`. A simulator and a real machine are two implementations of that one interface.
- **Disturbances are estimated, not ignored.** Whatever part of the vehicle's motion its own
  effectors and gravity do not explain is attributed to an outside force (a payload, wind, a
  push), and the control law leans against it.

| Path | What it is |
|---|---|
| `src/` | The kernel. `no_std`, no allocation, no `unsafe`. |
| `loader/` | Reads vehicle description files. |
| `python/` | Python binding, so the engine can drive a plant written in Python. |
| `vehicles/` | Vehicle descriptions. |
| `sim/` | Closed-loop and robustness tests in MuJoCo. |
| `examples/` | A quadrotor and a rocket flown by the same loop, in pure Rust. |

Frames follow ROS REP-103: the world is ENU (z up), the body is FLU (x forward, y left,
z up). Units are SI.

## Running it

Rust only:

```bash
cargo test --workspace
cargo run --release --example hover
```

In MuJoCo (needs [uv](https://docs.astral.sh/uv/)):

```bash
cd python && uvx maturin build --release --out ../target/wheels && cd ../sim
uv run --python 3.12 --with-requirements requirements.txt \
  --with ../target/wheels/april_engine-*.whl python mujoco_tests.py --out results
uv run --python 3.12 --with-requirements requirements.txt \
  --with ../target/wheels/april_engine-*.whl python robustness.py --out results/robustness
```

`mujoco_tests.py` flies every vehicle to its target in a perfect world. `robustness.py`
repeats that with sensor noise, sluggish effectors, 20 % of unannounced extra mass and a
steady push, alone and all together, and fails if a vehicle stops holding its target.
Add `--video` (with `MUJOCO_GL=egl`) for an mp4 of each run.

## Using it from Python

```python
import april_engine as ae

engine = ae.Engine(open("vehicles/quadrotor.toml").read(), target=(4.0, -2.0, 3.0))
engine.run(plant, ticks=20_000)
```

`plant` is any object with `measure()`, returning position, velocity, attitude as
`(w, x, y, z)` and angular velocity, and `actuate(commands, period)`.
