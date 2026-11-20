//! The engine's built-in plant: the vehicle as a single rigid body in free flight.
//!
//! There is no ground here, so it suits vehicles that fly. Anything that needs contact is
//! tested against an external simulator through the same [`Plant`] interface.

use core::convert::Infallible;

use crate::body::{RigidBody, State};
use crate::effector::{self, Effector};
use crate::engine::{Plant, Vehicle};
use crate::math::{Real, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct FreeBody<const N: usize> {
    body: RigidBody,
    effectors: [Effector; N],
    gravity: Vec3,
    /// The true state; a perfect sensor reads exactly this.
    pub state: State,
    /// A force (world frame, N) acting on the body from outside, such as wind.
    pub external_force: Vec3,
}

impl<const N: usize> FreeBody<N> {
    /// `vehicle` is the machine as it really is, which need not be how the engine was told
    /// it is.
    pub fn new(vehicle: &Vehicle<N>, gravity: Vec3, state: State) -> Self {
        Self {
            body: vehicle.body,
            effectors: vehicle.effectors,
            gravity,
            state,
            external_force: Vec3::zeros(),
        }
    }
}

impl<const N: usize> Plant<N> for FreeBody<N> {
    type Error = Infallible;

    fn measure(&mut self) -> Result<State, Infallible> {
        Ok(self.state)
    }

    fn actuate(&mut self, commands: &[Real; N], period: Real) -> Result<(), Infallible> {
        let wrench = effector::wrench(&self.effectors, commands);
        // A world-frame force on the centre of mass acts exactly like extra gravity.
        let field = self.gravity + self.external_force / self.body.mass();
        self.state = self.body.step(&self.state, &wrench, &field, period);
        Ok(())
    }
}
