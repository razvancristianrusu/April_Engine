//! Python binding for April Engine, so the engine's loop can drive a plant written in Python
//! (a simulator, typically). Vectors are 3-sequences, attitudes are `(w, x, y, z)`
//! body-to-world quaternions.

use ::april_engine as engine;
use april_engine_loader::{MAX_EFFECTORS, load};
use engine::{Disturbance, Disturbed, Plant, State, TickError, Vec3, math};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Triple = [f64; 3];
/// `(position, velocity, attitude, angular_velocity)`: position and velocity in the world
/// frame, angular velocity in the body frame.
type Measurement = (Triple, Triple, [f64; 4], Triple);

fn invalid(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn state((position, velocity, attitude, angular_velocity): Measurement) -> PyResult<State> {
    Ok(State {
        position: Vec3::from(position),
        velocity: Vec3::from(velocity),
        attitude: math::attitude_from_wxyz(attitude)
            .ok_or_else(|| PyValueError::new_err("attitude must be a finite non-zero quaternion"))?,
        angular_velocity: Vec3::from(angular_velocity),
    })
}

/// A Python object with `measure() -> (position, velocity, attitude, angular_velocity)` and
/// `actuate(commands, period)`, seen by the engine as a plant.
struct PythonPlant<'a, 'py> {
    object: &'a Bound<'py, PyAny>,
    effectors: usize,
}

impl Plant<MAX_EFFECTORS> for PythonPlant<'_, '_> {
    type Error = PyErr;

    fn measure(&mut self) -> PyResult<State> {
        state(self.object.call_method0("measure")?.extract()?)
    }

    fn actuate(&mut self, commands: &[f64; MAX_EFFECTORS], period: f64) -> PyResult<()> {
        let commands = commands[..self.effectors].to_vec();
        self.object.call_method1("actuate", (commands, period))?;
        Ok(())
    }
}

#[pyclass]
struct Engine {
    inner: engine::Engine<MAX_EFFECTORS>,
    name: String,
    effectors: usize,
}

#[pymethods]
impl Engine {
    /// `description` is the text of a TOML vehicle file, `target` a world-frame position.
    #[new]
    fn new(description: &str, target: Triple) -> PyResult<Self> {
        let loaded = load(description).map_err(invalid)?;
        Ok(Self {
            inner: loaded.engine(Vec3::from(target)).map_err(invalid)?,
            name: loaded.name,
            effectors: loaded.effectors,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Control period in seconds.
    #[getter]
    fn period(&self) -> f64 {
        self.inner.period()
    }

    /// Number of effectors, and so of commands per tick.
    #[getter]
    fn effectors(&self) -> usize {
        self.effectors
    }

    fn set_target(&mut self, target: Triple) {
        self.inner.set_target(Vec3::from(target));
    }

    /// The external force (world frame, N) the engine currently believes acts on the vehicle
    /// besides gravity and its own effectors.
    #[getter]
    fn disturbance(&self) -> Triple {
        (*self.inner.disturbance()).into()
    }

    /// The decision for one period: one command per effector for a vehicle found in this
    /// state. Call it once per period; it also updates the engine's disturbance estimate.
    fn step(
        &mut self,
        position: Triple,
        velocity: Triple,
        attitude: [f64; 4],
        angular_velocity: Triple,
    ) -> PyResult<Vec<f64>> {
        let state = state((position, velocity, attitude, angular_velocity))?;
        let commands = self.inner.step(&state).map_err(invalid)?;
        Ok(commands[..self.effectors].to_vec())
    }

    /// Runs the loop for `ticks` periods against `plant`: each tick calls `plant.measure()`
    /// and then `plant.actuate(commands, period)`.
    ///
    /// The keyword arguments put imperfect sensors and effectors between the engine and the
    /// plant: gaussian noise (standard deviation per axis, in m, m/s, rad and rad/s) on what
    /// is measured, and a first-order lag (time constant in s) on what is commanded. The
    /// same `seed` always gives the same noise.
    #[pyo3(signature = (plant, ticks, *, position_noise=0.0, velocity_noise=0.0,
        attitude_noise=0.0, angular_velocity_noise=0.0, effector_lag=0.0, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        plant: &Bound<'_, PyAny>,
        ticks: usize,
        position_noise: f64,
        velocity_noise: f64,
        attitude_noise: f64,
        angular_velocity_noise: f64,
        effector_lag: f64,
        seed: u64,
    ) -> PyResult<()> {
        let plant = PythonPlant {
            object: plant,
            effectors: self.effectors,
        };
        let disturbance = Disturbance {
            position_noise,
            velocity_noise,
            attitude_noise,
            angular_velocity_noise,
            effector_lag,
        };
        let mut plant = Disturbed::new(plant, disturbance, seed).map_err(invalid)?;
        for _ in 0..ticks {
            self.inner.tick(&mut plant).map_err(|error| match error {
                TickError::Plant(error) => error,
                TickError::Allocation(error) => invalid(error),
            })?;
        }
        Ok(())
    }
}

#[pymodule(name = "april_engine")]
fn bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Engine>()
}
