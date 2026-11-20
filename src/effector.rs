//! Effectors: the only way a vehicle acts on its own body.

use crate::math::{Axis, Real, Vec3, Wrench};

/// One scalar command and the wrench it produces: `command * unit_wrench`,
/// with `command` kept inside `[min, max]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Effector {
    pub unit_wrench: Wrench,
    pub min: Real,
    pub max: Real,
}

impl Effector {
    /// A force along `direction` applied at `position`, both in the body frame, with
    /// `position` measured from the centre of mass. The command is the force in newtons.
    ///
    /// Covers one component of a gimbaled thrust, a wheel's traction, a foot's contact force.
    pub fn force_at(position: Vec3, direction: Axis, min: Real, max: Real) -> Self {
        Self {
            unit_wrench: Wrench {
                force: direction.into_inner(),
                torque: position.cross(&direction),
            },
            min,
            max,
        }
    }

    /// A propeller thrusting along `axis` from `position`. The command is the thrust in newtons.
    ///
    /// `torque_per_thrust` (metres) is the reaction torque about `axis` per newton of thrust;
    /// its sign encodes the spin direction.
    pub fn rotor(position: Vec3, axis: Axis, torque_per_thrust: Real, max_thrust: Real) -> Self {
        let mut effector = Self::force_at(position, axis, 0.0, max_thrust);
        effector.unit_wrench.torque += axis.into_inner() * torque_per_thrust;
        effector
    }
}

/// The total wrench `effectors` produce for `commands`, paired in order.
pub fn wrench(effectors: &[Effector], commands: &[Real]) -> Wrench {
    let mut total = Wrench::default();
    for (effector, &command) in effectors.iter().zip(commands) {
        total.force += effector.unit_wrench.force * command;
        total.torque += effector.unit_wrench.torque * command;
    }
    total
}
