//! The engine proper: it owns the control loop, and everything else plugs into it.
//!
//! A [`Vehicle`] says what is being driven, a [`Plant`] is whatever is at the other end of the
//! wires (a simulator or the real machine), and [`Engine::tick`] is one turn of the loop:
//! measure, decide, actuate.

use core::fmt;

use crate::allocation::{AllocationError, Allocator};
use crate::body::{RigidBody, State};
use crate::control::Law;
use crate::effector::Effector;
use crate::math::{Real, Vec3, Wrench};

/// Everything the engine needs to know about a vehicle. Plain data: a new vehicle is a new
/// value of this type, not new code.
#[derive(Clone, Copy, Debug)]
pub struct Vehicle<const N: usize> {
    pub body: RigidBody,
    pub effectors: [Effector; N],
    /// Weight of each wrench axis in the allocation; 0 for axes the vehicle does not control.
    pub priority: Wrench,
    pub law: Law,
    /// How quickly (rad/s) the engine comes to believe in a steady force nobody told it
    /// about: a payload, wind, a slope. 0 turns that off and trusts the description as is.
    pub disturbance_bandwidth: Real,
}

/// What the engine drives. The same engine runs against a simulator and against hardware;
/// only the plant changes.
pub trait Plant<const N: usize> {
    type Error;

    /// The vehicle's state now.
    fn measure(&mut self) -> Result<State, Self::Error>;

    /// Applies `commands` and returns once `period` seconds have passed for the vehicle:
    /// a simulator advances its clock, hardware waits for the next tick.
    fn actuate(&mut self, commands: &[Real; N], period: Real) -> Result<(), Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineError {
    /// The control period must be finite and positive.
    InvalidPeriod,
    /// The disturbance bandwidth must be finite and non-negative.
    InvalidDisturbanceBandwidth,
    Allocation(AllocationError),
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPeriod => formatter.write_str("control period must be finite and positive"),
            Self::InvalidDisturbanceBandwidth => {
                formatter.write_str("disturbance bandwidth must be finite and non-negative")
            }
            Self::Allocation(error) => error.fmt(formatter),
        }
    }
}

impl core::error::Error for EngineError {}

/// Why a tick did not complete. Nothing was sent to the effectors on this tick unless the
/// failure came from [`Plant::actuate`] itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickError<E> {
    Plant(E),
    Allocation(AllocationError),
}

impl<E: fmt::Display> fmt::Display for TickError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plant(error) => write!(formatter, "plant failed: {error}"),
            Self::Allocation(error) => error.fmt(formatter),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> core::error::Error for TickError<E> {}

#[derive(Clone, Debug)]
pub struct Engine<const N: usize> {
    body: RigidBody,
    law: Law,
    allocator: Allocator<N>,
    gravity: Vec3,
    period: Real,
    target: Vec3,
    disturbance_bandwidth: Real,
    /// Where the vehicle's momentum should be by now if the description were the whole truth.
    expected_momentum: Option<Vec3>,
    disturbance: Vec3,
}

impl<const N: usize> Engine<N> {
    /// `period` is the control period in seconds, `gravity` and `target` are in the world frame.
    pub fn new(
        vehicle: &Vehicle<N>,
        period: Real,
        gravity: Vec3,
        target: Vec3,
    ) -> Result<Self, EngineError> {
        if !(period.is_finite() && period > 0.0) {
            return Err(EngineError::InvalidPeriod);
        }
        let bandwidth = vehicle.disturbance_bandwidth;
        if !(bandwidth.is_finite() && bandwidth >= 0.0) {
            return Err(EngineError::InvalidDisturbanceBandwidth);
        }
        Ok(Self {
            body: vehicle.body,
            law: vehicle.law,
            allocator: Allocator::new(vehicle.effectors, &vehicle.priority)
                .map_err(EngineError::Allocation)?,
            gravity,
            period,
            target,
            disturbance_bandwidth: bandwidth,
            expected_momentum: None,
            disturbance: Vec3::zeros(),
        })
    }

    pub fn period(&self) -> Real {
        self.period
    }

    pub fn target(&self) -> &Vec3 {
        &self.target
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    /// The external force (world frame, N) the engine currently believes acts on the vehicle
    /// besides gravity and its own effectors.
    pub fn disturbance(&self) -> &Vec3 {
        &self.disturbance
    }

    /// The decision for one period: effector commands for a vehicle found in `state`.
    ///
    /// Call it once per period: it also advances the engine's picture of what is pushing on
    /// the vehicle. Whatever part of the vehicle's momentum its own effectors and gravity do
    /// not account for is put down to an outside force, and the control law leans against it.
    /// The effectors enter as what they were actually asked for, limits included, so a
    /// saturated vehicle does not wind the estimate up.
    pub fn step(&mut self, state: &State) -> Result<[Real; N], AllocationError> {
        let mass = self.body.mass();
        let momentum = state.velocity * mass;
        let expected = self.expected_momentum.unwrap_or(momentum);
        let disturbance = (momentum - expected) * self.disturbance_bandwidth;

        let desired = self.law.wrench(
            &self.body,
            state,
            &self.target,
            &self.gravity,
            &disturbance,
        );
        let commands = self.allocator.allocate(&desired)?;

        let applied = state.attitude * self.allocator.wrench(&commands).force;
        self.expected_momentum =
            Some(expected + (applied + self.gravity * mass + disturbance) * self.period);
        self.disturbance = disturbance;
        Ok(commands)
    }

    /// One turn of the loop: measure, decide, actuate for one period. Returns the commands
    /// it sent.
    pub fn tick<P: Plant<N>>(&mut self, plant: &mut P) -> Result<[Real; N], TickError<P::Error>> {
        let state = plant.measure().map_err(TickError::Plant)?;
        let commands = self.step(&state).map_err(TickError::Allocation)?;
        plant
            .actuate(&commands, self.period)
            .map_err(TickError::Plant)?;
        Ok(commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{Pd, ThrustVectoring};
    use crate::math::Mat3;
    use crate::sim::FreeBody;

    const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.80665);

    fn quadrotor() -> Vehicle<4> {
        let rotor = |x: Real, y: Real, spin: Real| {
            Effector::rotor(Vec3::new(x, y, 0.0), Vec3::z_axis(), spin * 0.016, 8.0)
        };
        Vehicle {
            body: RigidBody::new(1.2, Mat3::from_diagonal(&Vec3::new(0.012, 0.012, 0.022)))
                .unwrap(),
            effectors: [
                rotor(0.14, 0.14, 1.0),
                rotor(-0.14, -0.14, 1.0),
                rotor(0.14, -0.14, -1.0),
                rotor(-0.14, 0.14, -1.0),
            ],
            priority: Wrench {
                force: Vec3::new(0.0, 0.0, 1.0),
                torque: Vec3::new(1.0, 1.0, 1.0),
            },
            law: Law::ThrustVectoring(ThrustVectoring {
                position: Pd::new(1.5, 1.0),
                attitude: Pd::new(12.0, 0.9),
                thrust_axis: Vec3::z_axis(),
                max_acceleration: 4.0,
            }),
            disturbance_bandwidth: 1.0,
        }
    }

    #[test]
    fn ticking_flies_the_vehicle_to_its_target_and_holds_it_there() {
        let vehicle = quadrotor();
        let target = Vec3::new(4.0, -2.0, 3.0);
        let mut engine = Engine::new(&vehicle, 1e-3, GRAVITY, target).unwrap();
        let mut plant = FreeBody::new(&vehicle, GRAVITY, State::at_rest(Vec3::zeros()));
        for _ in 0..12_000 {
            engine.tick(&mut plant).unwrap();
        }
        assert!((plant.state.position - target).norm() < 1e-3);
        assert!(plant.state.velocity.norm() < 1e-3);
    }

    #[test]
    fn retargeting_takes_effect_on_the_next_step() {
        let vehicle = quadrotor();
        let hover = State::at_rest(Vec3::zeros());
        let mut engine = Engine::new(&vehicle, 1e-3, GRAVITY, hover.position).unwrap();
        let holding: Real = engine.step(&hover).unwrap().iter().sum();
        engine.set_target(Vec3::new(0.0, 0.0, 5.0));
        let climbing: Real = engine.step(&hover).unwrap().iter().sum();
        assert!((holding - 1.2 * 9.80665).abs() < 1e-6);
        assert!(climbing > holding + 1.0);
    }

    #[test]
    fn unannounced_payload_and_wind_are_found_and_leaned_against() {
        let described = quadrotor();
        let mut real = described;
        real.body = RigidBody::new(1.44, *described.body.inertia()).unwrap();
        let wind = Vec3::new(2.0, 0.0, 0.0);
        let target = Vec3::new(1.0, 0.0, 2.0);

        let mut engine = Engine::new(&described, 1e-3, GRAVITY, target).unwrap();
        let mut plant = FreeBody::new(&real, GRAVITY, State::at_rest(Vec3::zeros()));
        plant.external_force = wind;
        for _ in 0..20_000 {
            engine.tick(&mut plant).unwrap();
        }

        assert!((plant.state.position - target).norm() < 1e-2);
        let payload_weight = GRAVITY * (1.44 - 1.2);
        assert!((engine.disturbance() - (wind + payload_weight)).norm() < 0.05);
    }

    #[test]
    fn unusable_period_is_refused() {
        let result = Engine::new(&quadrotor(), 0.0, GRAVITY, Vec3::zeros());
        assert_eq!(result.unwrap_err(), EngineError::InvalidPeriod);
    }
}
