#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
// #![deny(clippy::restriction)]
#![deny(clippy::cargo)]

extern crate ash;
extern crate bitflags;

pub mod dx12;
pub mod instance;
pub mod vulkan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Vulkan,
    DX12,
}
