//! Position hold for vehicles that push along one body axis and tilt to move sideways.

use core::f64::consts::PI;

use super::Pd;
use crate::body::{RigidBody, State};
use crate::math::{Axis, Quat, Real, Vec3, Wrench};

/// Position hold for any vehicle that can only push along one body axis and must therefore
/// tilt to move sideways: multirotors, gimbaled rockets, tail-sitters.
///
/// Rotation about the thrust axis is left free and only damped.
#[derive(Clone, Copy, Debug)]
pub struct ThrustVectoring {
    pub position: Pd,
    pub attitude: Pd,
    /// Body frame.
    pub thrust_axis: Axis,
    /// Cap on the acceleration the position loop may ask for (m/s²). Keeping it below gravity
    /// bounds the tilt under 45° however far away the target is.
    pub max_acceleration: Real,
}

impl ThrustVectoring {
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
        // Only the correction is capped: holding up the weight and leaning into whatever
        // pushes on the vehicle come on top, or a heavy load would use up all the authority.
        let acceleration = self
            .position
            .acceleration(&(target - state.position), &-state.velocity)
            .cap_magnitude(self.max_acceleration);
        let force = state
            .attitude
            .inverse_transform_vector(&((acceleration - gravity) * body.mass() - disturbance));

        // Without a force to point there is no preferred attitude; hold the current one.
        let tilt_error = Axis::try_new(force, Real::EPSILON)
            .map_or_else(Vec3::zeros, |direction| self.tilt_error(&direction));
        let angular_acceleration = self
            .attitude
            .acceleration(&tilt_error, &-state.angular_velocity);
        let momentum = body.inertia() * state.angular_velocity;

        Wrench {
            force: self.thrust_axis.into_inner() * force.dot(&self.thrust_axis),
            torque: body.inertia() * angular_acceleration
                + state.angular_velocity.cross(&momentum),
        }
    }

    /// Shortest rotation, as axis times angle in the body frame, that brings the thrust axis
    /// onto `direction` (body frame).
    fn tilt_error(&self, direction: &Axis) -> Vec3 {
        Quat::rotation_between_axis(&self.thrust_axis, direction).map_or_else(
            || {
                // Pointing exactly backwards: every perpendicular axis is equally short.
                let reference = if self.thrust_axis.x.abs() < 0.9 {
                    Vec3::x()
                } else {
                    Vec3::y()
                };
                self.thrust_axis.cross(&reference).normalize() * PI
            },
            |rotation| rotation.scaled_axis(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Mat3;

    const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.80665);

    fn law() -> ThrustVectoring {
        ThrustVectoring {
            position: Pd::new(1.5, 1.0),
            attitude: Pd::new(12.0, 0.9),
            thrust_axis: Vec3::z_axis(),
            max_acceleration: 4.0,
        }
    }

    #[test]
    fn hovering_on_target_asks_for_weight_and_nothing_else() {
        let body = RigidBody::new(2.0, Mat3::identity()).unwrap();
        let state = State::at_rest(Vec3::new(1.0, 2.0, 3.0));
        let wrench = law().wrench(&body, &state, &state.position, &GRAVITY, &Vec3::zeros());
        assert!((wrench.force - Vec3::new(0.0, 0.0, 2.0 * 9.80665)).norm() < 1e-9);
        assert!(wrench.torque.norm() < 1e-9);
    }

    #[test]
    fn inverted_vehicle_is_told_to_flip_upright() {
        let body = RigidBody::new(2.0, Mat3::identity()).unwrap();
        let mut state = State::at_rest(Vec3::zeros());
        state.attitude = Quat::from_scaled_axis(Vec3::x() * PI);
        let wrench = law().wrench(&body, &state, &state.position, &GRAVITY, &Vec3::zeros());
        assert!(wrench.torque.norm() > 1.0);
        assert!(wrench.torque.dot(&Vec3::z()).abs() < 1e-9);
    }

    #[test]
    fn known_side_force_is_leaned_into_without_giving_up_height() {
        let body = RigidBody::new(2.0, Mat3::identity()).unwrap();
        let state = State::at_rest(Vec3::zeros());
        let wind = Vec3::new(3.0, 0.0, 0.0);
        let wrench = law().wrench(&body, &state, &state.position, &GRAVITY, &wind);
        // Pushing back along -x means tipping the thrust axis that way: a turn about -y.
        assert!(wrench.torque.y < 0.0);
        assert!((wrench.force.z - 2.0 * 9.80665).abs() < 1e-9);
    }
}
