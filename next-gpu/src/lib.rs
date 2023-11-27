mod vulkan;

use vulkan::instance::VulkanInstance;

#[allow(dead_code)]
pub struct Instance {
    #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
    vulkan_instance: VulkanInstance,
}

impl Instance {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan_instance: VulkanInstance::new(),
        }
    }

    #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
    #[inline]
    pub fn vulkan(&self) -> &VulkanInstance {
        &self.vulkan_instance
    }
}
