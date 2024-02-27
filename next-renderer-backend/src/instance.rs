use crate::{vulkan::instance::VulkanInstance, Backend};

pub trait RawInstance {
    fn init(&mut self, desc: InstanceDescription);
}

pub struct InstanceDescription {}

pub struct Instance {
    #[allow(dead_code)]
    raw: Box<dyn RawInstance>,
}

impl Instance {
    pub fn new(backend: Backend, desc: InstanceDescription) -> Self {
        let mut raw = match backend {
            Backend::Vulkan => Box::new(VulkanInstance::new()),
            Backend::DX12 => unimplemented!(),
        };
        raw.init(desc);
        Self { raw }
    }
}
