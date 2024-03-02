use next_gpu_backend::Instance as RawInstance;
use next_gpu_backend::{api::instance::Description, dx12, vulkan, Backend};

#[allow(dead_code)]
pub struct Instance {
    vulkan: Option<vulkan::Instance>,
    dx12: Option<dx12::Instance>,
}

impl Instance {
    #[must_use]
    pub fn new(backend: Backend, desc: &Description) -> Self {
        match backend {
            Backend::Vulkan => Self {
                vulkan: Some(RawInstance::new(desc)),
                dx12: None,
            },
            Backend::DX12 => Self {
                vulkan: None,
                dx12: Some(RawInstance::new(desc)),
            },
        }
    }
}
