//! The imperfections every real machine has, as a layer around any [`Plant`].
//!
//! The engine is unaware of the layer: it measures and actuates as usual, and gets noisy
//! readings and sluggish effectors in return. What depends on the physics itself (a payload,
//! wind) belongs in the plant being wrapped.

use core::f64::consts::TAU;
use core::fmt;

use nalgebra::ComplexField;

use crate::body::State;
use crate::engine::Plant;
use crate::math::{Quat, Real, Vec3};

/// How imperfect the plant is. The default is a perfect one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Disturbance {
    /// Standard deviation of the noise on each axis of the measured position (m).
    pub position_noise: Real,
    /// Standard deviation of the noise on each axis of the measured velocity (m/s).
    pub velocity_noise: Real,
    /// Standard deviation of the measured attitude's error about each axis (rad).
    pub attitude_noise: Real,
    /// Standard deviation of the noise on each axis of the measured angular velocity (rad/s).
    pub angular_velocity_noise: Real,
    /// Time constant (s) with which each effector follows its command, starting from zero.
    /// 0 means effectors respond instantly.
    pub effector_lag: Real,
}

/// A disturbance magnitude was negative or not finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidDisturbance;

impl fmt::Display for InvalidDisturbance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("disturbance magnitudes must be finite and non-negative")
    }
}

impl core::error::Error for InvalidDisturbance {}

/// A plant seen through noisy sensors and sluggish effectors.
#[derive(Clone, Debug)]
pub struct Disturbed<P, const N: usize> {
    pub plant: P,
    disturbance: Disturbance,
    random: Random,
    applied: [Real; N],
}

impl<P: Plant<N>, const N: usize> Disturbed<P, N> {
    /// The same `seed` always produces the same noise.
    pub fn new(plant: P, disturbance: Disturbance, seed: u64) -> Result<Self, InvalidDisturbance> {
        let magnitudes = [
            disturbance.position_noise,
            disturbance.velocity_noise,
            disturbance.attitude_noise,
            disturbance.angular_velocity_noise,
            disturbance.effector_lag,
        ];
        if !magnitudes.iter().all(|value| value.is_finite() && *value >= 0.0) {
            return Err(InvalidDisturbance);
        }
        Ok(Self {
            plant,
            disturbance,
            random: Random::new(seed),
            applied: [0.0; N],
        })
    }
}

impl<P: Plant<N>, const N: usize> Plant<N> for Disturbed<P, N> {
    type Error = P::Error;

    fn measure(&mut self) -> Result<State, P::Error> {
        let truth = self.plant.measure()?;
        let noise = &self.disturbance;
        Ok(State {
            position: truth.position + self.random.vector(noise.position_noise),
            velocity: truth.velocity + self.random.vector(noise.velocity_noise),
            attitude: truth.attitude
                * Quat::from_scaled_axis(self.random.vector(noise.attitude_noise)),
            angular_velocity: truth.angular_velocity
                + self.random.vector(noise.angular_velocity_noise),
        })
    }

    fn actuate(&mut self, commands: &[Real; N], period: Real) -> Result<(), P::Error> {
        // Backward-Euler step of a first-order lag; with no lag it passes commands through.
        let blend = period / (self.disturbance.effector_lag + period);
        for (applied, command) in self.applied.iter_mut().zip(commands) {
            *applied += (command - *applied) * blend;
        }
        self.plant.actuate(&self.applied, period)
    }
}

/// Deterministic gaussian noise without an allocator or an operating system (splitmix64 for
/// the bits, Box-Muller for the shape).
#[derive(Clone, Debug)]
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> Real {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut bits = self.0;
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        bits ^= bits >> 31;
        ((bits >> 11) + 1) as Real / (1u64 << 53) as Real
    }

    fn gaussian(&mut self) -> Real {
        let radius = ComplexField::sqrt(-2.0 * ComplexField::ln(self.uniform()));
        radius * ComplexField::cos(TAU * self.uniform())
    }

    fn vector(&mut self, deviation: Real) -> Vec3 {
        if deviation == 0.0 {
            return Vec3::zeros();
        }
        Vec3::new(self.gaussian(), self.gaussian(), self.gaussian()) * deviation
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use super::*;

    /// A plant that stays put and remembers the last commands it was given.
    #[derive(Clone, Debug)]
    struct Bench {
        state: State,
        received: [Real; 1],
    }

    impl Bench {
        fn new() -> Self {
            Self {
                state: State::at_rest(Vec3::new(1.0, 2.0, 3.0)),
                received: [0.0],
            }
        }
    }

    impl Plant<1> for Bench {
        type Error = Infallible;

        fn measure(&mut self) -> Result<State, Infallible> {
            Ok(self.state)
        }

        fn actuate(&mut self, commands: &[Real; 1], _period: Real) -> Result<(), Infallible> {
            self.received = *commands;
            Ok(())
        }
    }

    #[test]
    fn perfect_disturbance_changes_nothing() {
        let mut plant = Disturbed::new(Bench::new(), Disturbance::default(), 7).unwrap();
        assert_eq!(plant.measure().unwrap(), Bench::new().state);
        plant.actuate(&[5.0], 1e-3).unwrap();
        assert_eq!(plant.plant.received, [5.0]);
    }

    #[test]
    fn lagging_effector_reaches_63_percent_of_a_step_after_one_time_constant() {
        let disturbance = Disturbance {
            effector_lag: 0.05,
            ..Disturbance::default()
        };
        let mut plant = Disturbed::new(Bench::new(), disturbance, 0).unwrap();
        for _ in 0..50 {
            plant.actuate(&[10.0], 1e-3).unwrap();
        }
        assert!((plant.plant.received[0] - 6.32).abs() < 0.05);
    }

    #[test]
    fn noise_has_the_requested_spread_and_no_bias() {
        let disturbance = Disturbance {
            position_noise: 0.2,
            ..Disturbance::default()
        };
        let mut plant = Disturbed::new(Bench::new(), disturbance, 42).unwrap();
        let samples = 20_000;
        let (mut sum, mut squares) = (0.0, 0.0);
        for _ in 0..samples {
            let error = plant.measure().unwrap().position.x - 1.0;
            sum += error;
            squares += error * error;
        }
        let mean: Real = sum / samples as Real;
        let deviation = ComplexField::sqrt(squares / samples as Real - mean * mean);
        assert!(mean.abs() < 0.01);
        assert!((deviation - 0.2).abs() < 0.01);
    }

    #[test]
    fn same_seed_gives_the_same_noise_and_another_seed_does_not() {
        let disturbance = Disturbance {
            velocity_noise: 0.1,
            ..Disturbance::default()
        };
        let reading = |seed| {
            Disturbed::new(Bench::new(), disturbance, seed)
                .unwrap()
                .measure()
                .unwrap()
        };
        assert_eq!(reading(3), reading(3));
        assert_ne!(reading(3), reading(4));
    }

    #[test]
    fn negative_magnitude_is_refused() {
        let disturbance = Disturbance {
            effector_lag: -0.1,
            ..Disturbance::default()
        };
        assert_eq!(
            Disturbed::new(Bench::new(), disturbance, 0).unwrap_err(),
            InvalidDisturbance
        );
    }
}
