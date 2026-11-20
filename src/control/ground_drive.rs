//! Position hold for vehicles that push along their forward axis and turn to move sideways.

use nalgebra::RealField;

use super::Pd;
use crate::body::{RigidBody, State};
use crate::math::{Axis, Real, Vec3, Wrench};

/// Position hold for any vehicle that pushes itself along its forward axis over a surface and
/// must turn to move sideways: differential-drive rovers, skid-steers, tracked vehicles.
///
/// The vehicle may push both ways, so a target behind it is reached in reverse rather than
/// by turning around. The component of the target along `up_axis` is ignored: the surface
/// decides the height, not the vehicle.
#[derive(Clone, Copy, Debug)]
pub struct GroundDrive {
    pub position: Pd,
    pub heading: Pd,
    /// Body frame, perpendicular to `up_axis`.
    pub forward_axis: Axis,
    /// Body frame: the axis the vehicle turns about.
    pub up_axis: Axis,
    /// Cap on the acceleration the position loop may ask for (m/s²). Keeping it below what the
    /// surface's grip allows is what stops the wheels from spinning.
    pub max_acceleration: Real,
    /// How close (m) the target has to be for steering to drop to half strength. Nearer than
    /// this the bearing to the target says more about sensor noise than about where to go, so
    /// steering fades out and the vehicle only closes the remaining gap along its forward axis.
    pub arrival_radius: Real,
}

impl GroundDrive {
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
        let forward = self.forward_axis.into_inner();
        let up = self.up_axis.into_inner();
        let left = up.cross(&forward);
        let to_body = |world: &Vec3| state.attitude.inverse_transform_vector(world);

        let error = to_body(&(target - state.position));
        let (ahead, aside) = (error.dot(&forward), error.dot(&left));
        let speed = to_body(&state.velocity).dot(&forward);
        let acceleration = self
            .position
            .acceleration(&(forward * ahead), &(forward * -speed))
            .cap_magnitude(self.max_acceleration);
        // The surface carries whatever acts across the forward axis; along it the vehicle has
        // to hold itself, against a slope and against anything pushing on it.
        let thrust = (acceleration.dot(&forward) - to_body(gravity).dot(&forward)) * body.mass()
            - to_body(disturbance).dot(&forward);

        // Turn whichever end is nearer towards the target.
        let facing = if ahead < 0.0 { -1.0 } else { 1.0 };
        let distance_squared = ahead * ahead + aside * aside;
        let radius_squared = self.arrival_radius * self.arrival_radius;
        let bearing = if distance_squared > 0.0 {
            RealField::atan2(facing * aside, facing * ahead) * distance_squared
                / (distance_squared + radius_squared)
        } else {
            0.0
        };
        let turn_rate = state.angular_velocity.dot(&up);
        let angular_acceleration = self
            .heading
            .acceleration(&(up * bearing), &(up * -turn_rate));
        let momentum = body.inertia() * state.angular_velocity;

        Wrench {
            force: forward * thrust,
            torque: body.inertia() * angular_acceleration
                + state.angular_velocity.cross(&momentum),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Mat3;

    const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.80665);

    fn law() -> GroundDrive {
        GroundDrive {
            position: Pd::new(1.2, 1.0),
            heading: Pd::new(6.0, 1.0),
            forward_axis: Vec3::x_axis(),
            up_axis: Vec3::z_axis(),
            max_acceleration: 1.5,
            arrival_radius: 0.03,
        }
    }

    fn wrench_towards(target: Vec3) -> Wrench {
        let body = RigidBody::new(12.0, Mat3::identity()).unwrap();
        law().wrench(&body, &State::at_rest(Vec3::zeros()), &target, &GRAVITY, &Vec3::zeros())
    }

    #[test]
    fn target_ahead_and_to_the_left_is_driven_at_while_turning_left() {
        let wrench = wrench_towards(Vec3::new(2.0, 1.0, 0.0));
        assert!(wrench.force.x > 0.0);
        assert!(wrench.torque.z > 0.0);
        assert!(wrench.force.y == 0.0 && wrench.force.z == 0.0);
    }

    #[test]
    fn target_behind_is_reached_in_reverse_without_turning_around() {
        let wrench = wrench_towards(Vec3::new(-2.0, 0.0, 0.0));
        assert!(wrench.force.x < 0.0);
        assert!(wrench.torque.norm() < 1e-9);
    }

    #[test]
    fn target_behind_and_to_the_left_swings_the_tail_towards_it() {
        // Reversing onto a target on the left means turning the nose to the right.
        let wrench = wrench_towards(Vec3::new(-2.0, 1.0, 0.0));
        assert!(wrench.force.x < 0.0);
        assert!(wrench.torque.z < 0.0);
    }

    #[test]
    fn steering_fades_out_as_the_target_gets_close() {
        // Same bearing both times; only the distance differs.
        let far = wrench_towards(Vec3::new(1.0, 1.0, 0.0)).torque.z;
        let near = wrench_towards(Vec3::new(0.003, 0.003, 0.0)).torque.z;
        assert!(far > 0.0 && near > 0.0);
        assert!(near < 0.05 * far);
    }

    #[test]
    fn known_push_from_behind_is_met_with_braking() {
        let body = RigidBody::new(12.0, Mat3::identity()).unwrap();
        let state = State::at_rest(Vec3::zeros());
        let push = Vec3::new(5.0, 0.0, 0.0);
        let wrench = law().wrench(&body, &state, &state.position, &GRAVITY, &push);
        assert!((wrench.force.x + 5.0).abs() < 1e-9);
    }

    #[test]
    fn slope_is_held_without_a_position_error() {
        let body = RigidBody::new(12.0, Mat3::identity()).unwrap();
        let mut state = State::at_rest(Vec3::zeros());
        // Nose pitched up by 0.2 rad: gravity now pulls the vehicle backwards along its axis.
        state.attitude = crate::math::Quat::from_scaled_axis(Vec3::y() * -0.2);
        let wrench = law().wrench(&body, &state, &state.position, &GRAVITY, &Vec3::zeros());
        let expected = 12.0 * 9.80665 * nalgebra::ComplexField::sin(0.2);
        assert!((wrench.force.x - expected).abs() < 1e-9);
    }
}
