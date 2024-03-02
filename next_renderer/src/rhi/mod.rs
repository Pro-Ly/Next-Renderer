use next_gpu_backend::{api::instance::Description, dx12, vulkan, Backend, Instance};

#[allow(dead_code)]
pub struct Instance1 {
    vulkan: Option<vulkan::Instance>,
    dx12: Option<dx12::Instance>,
}

impl Instance1 {
    #[must_use]
    pub fn new(backend: Backend, desc: &Description) -> Self {
        match backend {
            Backend::Vulkan => Self {
                vulkan: Some(Instance::new(desc)),
                dx12: None,
            },
            Backend::DX12 => Self {
                vulkan: None,
                dx12: Some(Instance::new(desc)),
            },
        }
    }
}
