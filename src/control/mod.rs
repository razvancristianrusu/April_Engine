//! Control laws: from where the vehicle is and where it should be to a desired body wrench.
//!
//! One law per way of getting around, not per vehicle: every law here serves any vehicle that
//! moves the same way, whatever its effectors are.

mod ground_drive;
mod thrust_vectoring;

pub use ground_drive::GroundDrive;
pub use thrust_vectoring::ThrustVectoring;

use crate::body::{RigidBody, State};
use crate::math::{Real, Vec3, Wrench};

/// Proportional-derivative law that outputs an acceleration, so its gains do not depend on the
/// vehicle's mass or inertia.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pd {
    pub stiffness: Real,
    pub damping: Real,
}

impl Pd {
    /// Gains for a second-order response with the given natural frequency (rad/s) and
    /// damping ratio.
    pub fn new(natural_frequency: Real, damping_ratio: Real) -> Self {
        Self {
            stiffness: natural_frequency * natural_frequency,
            damping: 2.0 * damping_ratio * natural_frequency,
        }
    }

    pub fn acceleration(&self, error: &Vec3, error_rate: &Vec3) -> Vec3 {
        error * self.stiffness + error_rate * self.damping
    }
}

/// The control law a vehicle runs, chosen by how it gets around.
#[derive(Clone, Copy, Debug)]
pub enum Law {
    ThrustVectoring(ThrustVectoring),
    GroundDrive(GroundDrive),
}

impl Law {
    /// The body wrench that drives the vehicle to rest at `target` (world frame), given the
    /// external force (world frame, N) currently believed to act on it besides gravity.
    pub fn wrench(
        &self,
        body: &RigidBody,
        state: &State,
        target: &Vec3,
        gravity: &Vec3,
        disturbance: &Vec3,
    ) -> Wrench {
        match self {
            Self::ThrustVectoring(law) => law.wrench(body, state, target, gravity, disturbance),
            Self::GroundDrive(law) => law.wrench(body, state, target, gravity, disturbance),
        }
    }
}
