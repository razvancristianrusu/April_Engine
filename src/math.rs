//! Scalar and spatial types shared by the whole engine.

/// The engine's scalar. Change it here to retarget the whole kernel to single precision.
pub type Real = f64;
pub type Vec3 = nalgebra::Vector3<Real>;
pub type Vec6 = nalgebra::Vector6<Real>;
pub type Mat3 = nalgebra::Matrix3<Real>;
/// Rotation taking body-frame vectors to the world frame.
pub type Quat = nalgebra::UnitQuaternion<Real>;
pub type Axis = nalgebra::Unit<Vec3>;

/// An attitude from the `(w, x, y, z)` components most simulators and middlewares report.
/// `None` if they are not finite or all zero; anything else is normalised.
pub fn attitude_from_wxyz([w, x, y, z]: [Real; 4]) -> Option<Quat> {
    Quat::try_new(nalgebra::Quaternion::new(w, x, y, z), Real::EPSILON)
}

/// Force and torque about the centre of mass, both expressed in the body frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wrench {
    pub force: Vec3,
    pub torque: Vec3,
}

impl Wrench {
    /// Stacked as `[force, torque]`.
    pub fn to_vector(&self) -> Vec6 {
        Vec6::new(
            self.force.x,
            self.force.y,
            self.force.z,
            self.torque.x,
            self.torque.y,
            self.torque.z,
        )
    }

    pub fn is_finite(&self) -> bool {
        self.force.iter().chain(self.torque.iter()).all(|value| value.is_finite())
    }
}
