use bitflags::bitflags;

use crate::{dx12::instance::DX12Instance, vulkan::instance::VulkanInstance, Backend};
pub trait RawInstance {}

pub struct InstanceDescription {
    flags: InstanceFlags,
}

impl InstanceDescription {
    #[must_use]
    pub fn flags(&self) -> &InstanceFlags {
        &self.flags
    }
}

impl Default for InstanceDescription {
    #[must_use]
    fn default() -> Self {
        Self {
            flags: InstanceFlags::empty(),
        }
    }
}

bitflags! {
    pub struct InstanceFlags: u32 {
        const DEBUG = 1 << 0;
        const VALIDATION = 1 << 1;
    }
}

pub struct Instance {
    #[allow(dead_code)]
    raw: Box<dyn RawInstance>,
}

impl Instance {
    #[must_use]
    pub fn new(backend: Backend, desc: &InstanceDescription) -> Self {
        let raw: Box<dyn RawInstance> = match backend {
            Backend::Vulkan => Box::new(VulkanInstance::new(desc)),
            Backend::DX12 => Box::new(DX12Instance::new(desc)),
        };
        // raw.init(desc);
        Self { raw }
    }
}
