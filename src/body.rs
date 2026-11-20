//! The rigid body every vehicle is reduced to, and its equations of motion.

use core::fmt;

use crate::math::{Mat3, Quat, Real, Vec3, Wrench};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyError {
    /// Mass must be finite and positive.
    InvalidMass,
    /// Inertia must be symmetric positive definite.
    InvalidInertia,
}

impl fmt::Display for BodyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidMass => "mass must be finite and positive",
            Self::InvalidInertia => "inertia must be symmetric positive definite",
        })
    }
}

impl core::error::Error for BodyError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    /// World frame.
    pub position: Vec3,
    /// World frame.
    pub velocity: Vec3,
    pub attitude: Quat,
    /// Body frame.
    pub angular_velocity: Vec3,
}

impl State {
    /// Motionless and level at `position`.
    pub fn at_rest(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::zeros(),
            attitude: Quat::identity(),
            angular_velocity: Vec3::zeros(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    mass: Real,
    inertia: Mat3,
    inertia_inverse: Mat3,
}

impl RigidBody {
    /// `inertia` is taken about the centre of mass, in the body frame.
    pub fn new(mass: Real, inertia: Mat3) -> Result<Self, BodyError> {
        if !(mass.is_finite() && mass > 0.0) {
            return Err(BodyError::InvalidMass);
        }
        if inertia != inertia.transpose() {
            return Err(BodyError::InvalidInertia);
        }
        let inertia_inverse = inertia
            .cholesky()
            .ok_or(BodyError::InvalidInertia)?
            .inverse();
        Ok(Self {
            mass,
            inertia,
            inertia_inverse,
        })
    }

    pub fn mass(&self) -> Real {
        self.mass
    }

    pub fn inertia(&self) -> &Mat3 {
        &self.inertia
    }

    /// Advances `state` by `dt` under `wrench` and a uniform `gravity` acceleration (world frame).
    ///
    /// Semi-implicit Euler: velocities are updated first and then carry the pose, which keeps
    /// the integration stable at control-loop step sizes.
    pub fn step(&self, state: &State, wrench: &Wrench, gravity: &Vec3, dt: Real) -> State {
        let acceleration = state.attitude * wrench.force / self.mass + gravity;
        let momentum = self.inertia * state.angular_velocity;
        let angular_acceleration =
            self.inertia_inverse * (wrench.torque - state.angular_velocity.cross(&momentum));

        let velocity = state.velocity + acceleration * dt;
        let angular_velocity = state.angular_velocity + angular_acceleration * dt;
        State {
            position: state.position + velocity * dt,
            velocity,
            attitude: state.attitude * Quat::from_scaled_axis(angular_velocity * dt),
            angular_velocity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: Real = 1e-4;
    const STEPS: usize = 10_000;

    #[test]
    fn free_fall_matches_the_closed_form() {
        let body = RigidBody::new(3.0, Mat3::identity()).unwrap();
        let gravity = Vec3::new(0.0, 0.0, -9.80665);
        let mut state = State::at_rest(Vec3::zeros());
        for _ in 0..STEPS {
            state = body.step(&state, &Wrench::default(), &gravity, DT);
        }
        let elapsed = DT * STEPS as Real;
        assert!((state.velocity.z - gravity.z * elapsed).abs() < 1e-9);
        assert!((state.position.z - 0.5 * gravity.z * elapsed * elapsed).abs() < 1e-3);
    }

    #[test]
    fn torque_free_tumbling_conserves_angular_momentum() {
        let inertia = Mat3::from_diagonal(&Vec3::new(1.0, 2.0, 3.0));
        let body = RigidBody::new(1.0, inertia).unwrap();
        let mut state = State::at_rest(Vec3::zeros());
        state.angular_velocity = Vec3::new(1.0, 0.5, -0.3);
        let world_momentum =
            |state: &State| state.attitude * (inertia * state.angular_velocity);

        let before = world_momentum(&state);
        for _ in 0..STEPS {
            state = body.step(&state, &Wrench::default(), &Vec3::zeros(), DT);
        }
        assert!((world_momentum(&state) - before).norm() < 1e-3 * before.norm());
    }
}
