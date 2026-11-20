//! Control allocation: from the wrench a control law wants to the commands effectors can give.

use core::fmt;

use nalgebra::{Matrix6, SMatrix, SVector};

use crate::effector::Effector;
use crate::math::{Real, Vec6, Wrench};

/// Regularisation of the least-squares solve, relative to the effectiveness matrix's scale.
/// Keeps the solve well-posed for axes the effectors cannot reach at all.
const RELATIVE_DAMPING: Real = 1e-9;
const MIN_DAMPING: Real = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationError {
    /// The effector's unit wrench is not finite, or its limits are not ordered.
    InvalidEffector { index: usize },
    /// Priorities must be finite and non-negative.
    InvalidPriority,
    /// The requested wrench has no finite solution; nothing safe can be commanded.
    NonFinite,
}

impl fmt::Display for AllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEffector { index } => write!(
                formatter,
                "effector {index} has a non-finite unit wrench or unordered limits"
            ),
            Self::InvalidPriority => {
                formatter.write_str("axis priorities must be finite and non-negative")
            }
            Self::NonFinite => {
                formatter.write_str("requested wrench has no finite effector solution")
            }
        }
    }
}

impl core::error::Error for AllocationError {}

/// Maps a desired body wrench onto `N` effectors.
#[derive(Clone, Debug)]
pub struct Allocator<const N: usize> {
    effectors: [Effector; N],
    /// Effectiveness matrix (one unit wrench per column) with rows scaled by `priority`.
    weighted: SMatrix<Real, 6, N>,
    priority: Vec6,
    damping: Real,
}

impl<const N: usize> Allocator<N> {
    /// `priority` weighs each wrench axis in the least-squares fit. It sets what gives way when
    /// the effectors cannot deliver everything, and it puts forces (N) and torques (N·m) on a
    /// common scale. An axis the vehicle does not control gets 0.
    pub fn new(effectors: [Effector; N], priority: &Wrench) -> Result<Self, AllocationError> {
        let priority = priority.to_vector();
        if !priority.iter().all(|weight| weight.is_finite() && *weight >= 0.0) {
            return Err(AllocationError::InvalidPriority);
        }
        let mut weighted = SMatrix::<Real, 6, N>::zeros();
        for (index, effector) in effectors.iter().enumerate() {
            if !(effector.unit_wrench.is_finite() && effector.min <= effector.max) {
                return Err(AllocationError::InvalidEffector { index });
            }
            weighted.set_column(
                index,
                &priority.component_mul(&effector.unit_wrench.to_vector()),
            );
        }
        Ok(Self {
            effectors,
            weighted,
            priority,
            damping: (weighted.norm_squared() * RELATIVE_DAMPING).max(MIN_DAMPING),
        })
    }

    /// The wrench the effectors produce for `commands`.
    pub fn wrench(&self, commands: &[Real; N]) -> Wrench {
        crate::effector::wrench(&self.effectors, commands)
    }

    /// Commands, each inside its effector's limits, whose wrench is closest to `desired`.
    ///
    /// Redistributed pseudo-inverse: solve the minimum-norm least-squares problem, pin every
    /// effector that lands outside its limits at that limit, and solve again for the rest
    /// against what is still missing.
    pub fn allocate(&self, desired: &Wrench) -> Result<[Real; N], AllocationError> {
        let target = self.priority.component_mul(&desired.to_vector());
        let mut commands = SVector::<Real, N>::zeros();
        let mut free = [true; N];

        // A pass that does not finish pins at least one effector, so N passes always suffice.
        for _ in 0..N {
            let mut active = self.weighted;
            let mut residual = target;
            for index in 0..N {
                if !free[index] {
                    residual -= active.column(index) * commands[index];
                    active.column_mut(index).fill(0.0);
                }
            }
            let gram = active * active.transpose() + Matrix6::identity() * self.damping;
            let solution = active.tr_mul(
                &gram
                    .cholesky()
                    .ok_or(AllocationError::NonFinite)?
                    .solve(&residual),
            );

            let mut pinned = false;
            for index in 0..N {
                if free[index] {
                    let effector = &self.effectors[index];
                    commands[index] = solution[index].clamp(effector.min, effector.max);
                    if commands[index] != solution[index] {
                        free[index] = false;
                        pinned = true;
                    }
                }
            }
            if !pinned {
                break;
            }
        }

        if commands.iter().all(|command| command.is_finite()) {
            Ok(commands.into())
        } else {
            Err(AllocationError::NonFinite)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Axis, Vec3};

    fn up() -> Axis {
        Vec3::z_axis()
    }

    fn quadrotor() -> Allocator<4> {
        let rotor = |x: Real, y: Real, spin: Real| {
            Effector::rotor(Vec3::new(x, y, 0.0), up(), spin * 0.016, 8.0)
        };
        Allocator::new(
            [
                rotor(0.14, 0.14, 1.0),
                rotor(-0.14, -0.14, 1.0),
                rotor(0.14, -0.14, -1.0),
                rotor(-0.14, 0.14, -1.0),
            ],
            &Wrench {
                force: Vec3::new(0.0, 0.0, 1.0),
                torque: Vec3::new(1.0, 1.0, 1.0),
            },
        )
        .unwrap()
    }

    #[test]
    fn reachable_wrench_is_produced_exactly() {
        let allocator = quadrotor();
        let desired = Wrench {
            force: Vec3::new(0.0, 0.0, 12.0),
            torque: Vec3::new(0.1, -0.2, 0.05),
        };
        let achieved = allocator.wrench(&allocator.allocate(&desired).unwrap());
        assert!((achieved.to_vector() - desired.to_vector()).norm() < 1e-6);
    }

    #[test]
    fn saturated_effector_hands_its_share_to_the_others() {
        let pusher =
            |max: Real| Effector::force_at(Vec3::zeros(), Vec3::x_axis(), 0.0, max);
        let allocator = Allocator::new(
            [pusher(1.0), pusher(10.0)],
            &Wrench {
                force: Vec3::new(1.0, 1.0, 1.0),
                torque: Vec3::new(1.0, 1.0, 1.0),
            },
        )
        .unwrap();
        let desired = Wrench {
            force: Vec3::new(6.0, 0.0, 0.0),
            torque: Vec3::zeros(),
        };
        let commands = allocator.allocate(&desired).unwrap();
        assert!((commands[0] - 1.0).abs() < 1e-6);
        assert!((commands[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn unprioritised_axis_does_not_dilute_the_controlled_ones() {
        // Gimbaled thrust at the tail: side force and pitch torque come from the same command,
        // so the torque can only be exact if the side-force axis is given up.
        let tail = Vec3::new(0.0, 0.0, -1.0);
        let allocator = Allocator::new(
            [
                Effector::force_at(tail, Vec3::x_axis(), -50.0, 50.0),
                Effector::force_at(tail, Vec3::y_axis(), -50.0, 50.0),
                Effector::force_at(tail, up(), 0.0, 500.0),
            ],
            &Wrench {
                force: Vec3::new(0.0, 0.0, 1.0),
                torque: Vec3::new(1.0, 1.0, 0.0),
            },
        )
        .unwrap();
        let desired = Wrench {
            force: Vec3::new(0.0, 0.0, 300.0),
            torque: Vec3::new(20.0, -10.0, 0.0),
        };
        let achieved = allocator.wrench(&allocator.allocate(&desired).unwrap());
        assert!((achieved.torque - desired.torque).norm() < 1e-6);
        assert!((achieved.force.z - desired.force.z).abs() < 1e-6);
    }

    #[test]
    fn non_finite_request_never_reaches_the_effectors() {
        let desired = Wrench {
            force: Vec3::new(0.0, 0.0, Real::NAN),
            torque: Vec3::zeros(),
        };
        assert_eq!(
            quadrotor().allocate(&desired),
            Err(AllocationError::NonFinite)
        );
    }
}
