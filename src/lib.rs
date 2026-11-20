//! April Engine: one control kernel for anything that moves.
//!
//! Every vehicle is reduced to the same three parts:
//!
//! 1. a [`RigidBody`] carrying a [`State`];
//! 2. a set of [`Effector`]s, each turning one scalar command into a wrench on that body
//!    (a rotor, one component of a gimbaled thrust, a wheel's traction, a foot's contact force);
//! 3. a control [`Law`] that asks for a body [`Wrench`].
//!
//! The [`Allocator`] closes the gap between 3 and 2: it finds the effector commands that best
//! produce the requested wrench inside the effectors' limits. A new kind of vehicle is a new
//! effector layout and a new control law; the kernel underneath does not change.
//!
//! Those three parts make a [`Vehicle`], which is plain data. The [`Engine`] takes one and
//! owns the loop: each [`Engine::tick`] measures, decides and actuates through a [`Plant`],
//! which is a simulator or the real machine. Nothing else differs between the two.
//!
//! Frames follow ROS REP-103: the world is ENU (z up), the body is FLU (x forward, y left, z up).
//! Units are SI.
#![no_std]
#![forbid(unsafe_code)]

pub mod allocation;
pub mod body;
pub mod control;
pub mod disturbance;
pub mod effector;
pub mod engine;
pub mod math;
pub mod sim;

pub use allocation::{AllocationError, Allocator};
pub use body::{BodyError, RigidBody, State};
pub use control::{GroundDrive, Law, Pd, ThrustVectoring};
pub use disturbance::{Disturbance, Disturbed, InvalidDisturbance};
pub use effector::Effector;
pub use engine::{Engine, EngineError, Plant, TickError, Vehicle};
pub use math::{Axis, Mat3, Quat, Real, Vec3, Vec6, Wrench};
pub use sim::FreeBody;
