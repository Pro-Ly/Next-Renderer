#![deny(clippy::pedantic)]
// #![deny(clippy::restriction)]
#![deny(clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

extern crate next_gpu_backend;
extern crate winit;

pub mod rhi;
pub mod window;

pub use next_gpu_backend::*;
