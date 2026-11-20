//! Builds vehicles from description files, so that a new vehicle is a new file.
//!
//! The kernel sizes its allocator at compile time, while a file decides the effector count at
//! run time. Loaded vehicles therefore all have [`MAX_EFFECTORS`] slots, with the unused ones
//! filled by effectors that can do nothing.

use std::fmt;

use april_engine::{
    Axis, BodyError, Effector, Engine, EngineError, GroundDrive, Law, Mat3, Pd, RigidBody,
    ThrustVectoring, Vec3, Vehicle, Wrench,
};
use serde::Deserialize;

pub const MAX_EFFECTORS: usize = 16;

const STANDARD_GRAVITY: [f64; 3] = [0.0, 0.0, -9.80665];

#[derive(Debug)]
pub enum LoadError {
    Syntax(toml::de::Error),
    /// The named axis is zero or not finite.
    InvalidAxis(&'static str),
    TooManyEffectors(usize),
    Body(BodyError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(error) => error.fmt(formatter),
            Self::InvalidAxis(name) => write!(formatter, "{name} must be a finite non-zero vector"),
            Self::TooManyEffectors(count) => write!(
                formatter,
                "{count} effectors described, at most {MAX_EFFECTORS} are supported"
            ),
            Self::Body(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for LoadError {}

/// A vehicle as read from its description.
#[derive(Clone, Debug)]
pub struct Loaded {
    pub name: String,
    pub vehicle: Vehicle<MAX_EFFECTORS>,
    /// How many of the vehicle's effector slots are real; they come first.
    pub effectors: usize,
    /// Control period in seconds.
    pub period: f64,
    pub gravity: Vec3,
}

impl Loaded {
    pub fn engine(&self, target: Vec3) -> Result<Engine<MAX_EFFECTORS>, EngineError> {
        Engine::new(&self.vehicle, self.period, self.gravity, target)
    }
}

/// Parses a TOML vehicle description.
pub fn load(text: &str) -> Result<Loaded, LoadError> {
    let description: Description = toml::from_str(text).map_err(LoadError::Syntax)?;
    let count = description.effector.len();
    if count > MAX_EFFECTORS {
        return Err(LoadError::TooManyEffectors(count));
    }
    let idle = Effector {
        unit_wrench: Wrench::default(),
        min: 0.0,
        max: 0.0,
    };
    let mut effectors = [idle; MAX_EFFECTORS];
    for (slot, described) in effectors.iter_mut().zip(&description.effector) {
        *slot = described.build()?;
    }
    let inertia = Mat3::from_fn(|row, column| description.body.inertia[row][column]);
    let [fx, fy, fz, tx, ty, tz] = description.allocation.priority;
    Ok(Loaded {
        name: description.name,
        vehicle: Vehicle {
            body: RigidBody::new(description.body.mass, inertia).map_err(LoadError::Body)?,
            effectors,
            priority: Wrench {
                force: Vec3::new(fx, fy, fz),
                torque: Vec3::new(tx, ty, tz),
            },
            law: description.law.build()?,
            disturbance_bandwidth: description.disturbance_bandwidth,
        },
        effectors: count,
        period: description.period,
        gravity: Vec3::from(description.gravity),
    })
}

fn axis(name: &'static str, value: [f64; 3]) -> Result<Axis, LoadError> {
    Axis::try_new(Vec3::from(value), f64::EPSILON).ok_or(LoadError::InvalidAxis(name))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Description {
    name: String,
    period: f64,
    disturbance_bandwidth: f64,
    #[serde(default = "standard_gravity")]
    gravity: [f64; 3],
    body: BodyDescription,
    #[serde(default)]
    effector: Vec<EffectorDescription>,
    allocation: AllocationDescription,
    law: LawDescription,
}

fn standard_gravity() -> [f64; 3] {
    STANDARD_GRAVITY
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BodyDescription {
    mass: f64,
    inertia: [[f64; 3]; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AllocationDescription {
    priority: [f64; 6],
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum EffectorDescription {
    Force {
        position: [f64; 3],
        direction: [f64; 3],
        min: f64,
        max: f64,
    },
    Rotor {
        position: [f64; 3],
        axis: [f64; 3],
        torque_per_thrust: f64,
        max: f64,
    },
}

impl EffectorDescription {
    fn build(&self) -> Result<Effector, LoadError> {
        Ok(match *self {
            Self::Force {
                position,
                direction,
                min,
                max,
            } => Effector::force_at(Vec3::from(position), axis("direction", direction)?, min, max),
            Self::Rotor {
                position,
                axis: thrust_axis,
                torque_per_thrust,
                max,
            } => Effector::rotor(
                Vec3::from(position),
                axis("axis", thrust_axis)?,
                torque_per_thrust,
                max,
            ),
        })
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gains {
    natural_frequency: f64,
    damping_ratio: f64,
}

impl Gains {
    fn build(self) -> Pd {
        Pd::new(self.natural_frequency, self.damping_ratio)
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum LawDescription {
    ThrustVectoring {
        position: Gains,
        attitude: Gains,
        thrust_axis: [f64; 3],
        max_acceleration: f64,
    },
    GroundDrive {
        position: Gains,
        heading: Gains,
        forward_axis: [f64; 3],
        up_axis: [f64; 3],
        max_acceleration: f64,
        arrival_radius: f64,
    },
}

impl LawDescription {
    fn build(&self) -> Result<Law, LoadError> {
        Ok(match *self {
            Self::ThrustVectoring {
                position,
                attitude,
                thrust_axis,
                max_acceleration,
            } => Law::ThrustVectoring(ThrustVectoring {
                position: position.build(),
                attitude: attitude.build(),
                thrust_axis: axis("thrust_axis", thrust_axis)?,
                max_acceleration,
            }),
            Self::GroundDrive {
                position,
                heading,
                forward_axis,
                up_axis,
                max_acceleration,
                arrival_radius,
            } => Law::GroundDrive(GroundDrive {
                position: position.build(),
                heading: heading.build(),
                forward_axis: axis("forward_axis", forward_axis)?,
                up_axis: axis("up_axis", up_axis)?,
                max_acceleration,
                arrival_radius,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use april_engine::State;

    const QUADROTOR: &str = include_str!("../../vehicles/quadrotor.toml");

    #[test]
    fn every_shipped_vehicle_loads_into_a_working_engine() {
        for text in [
            QUADROTOR,
            include_str!("../../vehicles/rocket.toml"),
            include_str!("../../vehicles/rover.toml"),
        ] {
            let loaded = load(text).unwrap();
            let mut engine = loaded.engine(Vec3::zeros()).unwrap();
            engine.step(&State::at_rest(Vec3::zeros())).unwrap();
        }
    }

    #[test]
    fn described_vehicle_hovers_on_its_own_weight_using_only_its_real_effectors() {
        let loaded = load(QUADROTOR).unwrap();
        let hover = State::at_rest(Vec3::new(1.0, 2.0, 3.0));
        let commands = loaded.engine(hover.position).unwrap().step(&hover).unwrap();
        let (real, idle) = commands.split_at(loaded.effectors);
        assert_eq!(loaded.effectors, 4);
        assert!(real.iter().all(|thrust| (thrust - 1.2 * 9.80665 / 4.0).abs() < 1e-6));
        assert!(idle.iter().all(|&command| command == 0.0));
    }

    #[test]
    fn misspelt_key_is_rejected_instead_of_ignored() {
        let text = QUADROTOR.replace("max_acceleration", "max_acelleration");
        assert!(matches!(load(&text), Err(LoadError::Syntax(_))));
    }
}
