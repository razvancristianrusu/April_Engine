//! One engine, two vehicles: a quadrotor and a rocket balancing on a gimbaled engine.
//! Only the vehicle description differs; the loop is the engine's.
//!
//! Run with `cargo run --example hover`.

use std::error::Error;

use april_engine::{
    Effector, Engine, FreeBody, Law, Mat3, Pd, Real, RigidBody, State, ThrustVectoring, Vec3,
    Vehicle, Wrench,
};

const PERIOD: Real = 1e-3;
const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.80665);
const DURATION: Real = 20.0;
const ARRIVAL_TOLERANCE: Real = 0.02;

fn quadrotor() -> Result<Vehicle<4>, Box<dyn Error>> {
    let rotor = |x: Real, y: Real, spin: Real| {
        Effector::rotor(Vec3::new(x, y, 0.0), Vec3::z_axis(), spin * 0.016, 8.0)
    };
    Ok(Vehicle {
        body: RigidBody::new(
            1.2,
            Mat3::from_diagonal(&Vec3::new(0.012, 0.012, 0.022)),
        )?,
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
    })
}

/// A hopper on one engine gimbaled 8° about its tail. The gimbaled thrust is commanded as its
/// three force components; it has no authority over roll.
fn rocket() -> Result<Vehicle<3>, Box<dyn Error>> {
    let max_thrust: Real = 600.0;
    let side = max_thrust * 8.0_f64.to_radians().sin();
    let gimbal = Vec3::new(0.0, 0.0, -1.0);
    Ok(Vehicle {
        body: RigidBody::new(30.0, Mat3::from_diagonal(&Vec3::new(10.0, 10.0, 0.5)))?,
        effectors: [
            Effector::force_at(gimbal, Vec3::x_axis(), -side, side),
            Effector::force_at(gimbal, Vec3::y_axis(), -side, side),
            Effector::force_at(gimbal, Vec3::z_axis(), 0.2 * max_thrust, max_thrust),
        ],
        priority: Wrench {
            force: Vec3::new(0.0, 0.0, 1.0),
            torque: Vec3::new(1.0, 1.0, 0.0),
        },
        law: Law::ThrustVectoring(ThrustVectoring {
            position: Pd::new(0.8, 1.0),
            attitude: Pd::new(5.0, 1.0),
            thrust_axis: Vec3::z_axis(),
            max_acceleration: 2.0,
        }),
        disturbance_bandwidth: 1.0,
    })
}

fn fly<const N: usize>(
    name: &str,
    vehicle: &Vehicle<N>,
    target: Vec3,
) -> Result<(), Box<dyn Error>> {
    let mut engine = Engine::new(vehicle, PERIOD, GRAVITY, target)?;
    let mut plant = FreeBody::new(vehicle, GRAVITY, State::at_rest(Vec3::zeros()));
    let ticks = (DURATION / PERIOD).round() as usize;
    let report_every = ticks / 10;

    println!("\n{name}");
    println!("  target ({:.1}, {:.1}, {:.1}) m", target.x, target.y, target.z);
    println!("  {:>5} {:>24} {:>6}  commands", "t s", "position m", "tilt°");
    for tick in 0..=ticks {
        let state = plant.state;
        let commands = engine.tick(&mut plant)?;
        if tick % report_every == 0 {
            println!(
                "  {:5.1} ({:6.2}, {:6.2}, {:6.2}) {:6.1}  {:7.2?}",
                tick as Real * PERIOD,
                state.position.x,
                state.position.y,
                state.position.z,
                (state.attitude * Vec3::z()).z.acos().to_degrees(),
                commands,
            );
        }
    }

    let miss = (plant.state.position - target).norm();
    let speed = plant.state.velocity.norm();
    println!("  final miss {miss:.4} m, speed {speed:.4} m/s");
    if miss > ARRIVAL_TOLERANCE || speed > ARRIVAL_TOLERANCE {
        return Err(format!("{name} did not settle on its target").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    fly(
        "quadrotor (4 rotors, command = thrust per rotor, N)",
        &quadrotor()?,
        Vec3::new(4.0, -2.0, 3.0),
    )?;
    fly(
        "rocket (gimbaled engine, command = thrust components x y z, N)",
        &rocket()?,
        Vec3::new(6.0, 3.0, 20.0),
    )?;
    Ok(())
}
