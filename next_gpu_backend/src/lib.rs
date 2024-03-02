#![deny(clippy::pedantic)]
// #![deny(clippy::restriction)]
#![deny(clippy::cargo)]

extern crate ash;
extern crate bitflags;

use api::instance::Description;

pub mod api;
pub mod dx12;
pub mod vulkan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Vulkan,
    DX12,
}

pub trait Api: Sized {
    type Instance: Instance<Self>;
}

pub trait Instance<TApi: Api>: Sized {
    fn new(desc: &Description) -> Self;
}
