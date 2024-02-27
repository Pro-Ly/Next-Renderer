use std::ffi::{c_char, CStr};

use crate::instance::{InstanceDescription, RawInstance};

// use ash::extensions::ext::DebugUtils;

#[allow(dead_code)]
pub struct VulkanInstance {
    entry: ash::Entry,
}
#[allow(unused_variables)]
#[allow(clippy::new_without_default)]
impl VulkanInstance {
    pub fn new() -> Self {
        let entry = ash::Entry::linked();

        // if debug
        let layer_names =
            [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
        #[allow(dead_code)]
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        // let mut extension_names =
        //     ash_window::enumerate_required_extensions(window.raw_display_handle())
        //         .unwrap()
        //         .to_vec();
        // extension_names.push(DebugUtils::name().as_ptr());

        // #[cfg(any(target_os = "macos", target_os = "ios"))]
        // {
        //     extension_names.push(KhrPortabilityEnumerationFn::NAME.as_ptr());
        //     // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
        //     extension_names.push(KhrGetPhysicalDeviceProperties2Fn::NAME.as_ptr());
        // }

        // let appinfo = vk::ApplicationInfo::default()
        //     .application_name(app_name)
        //     .application_version(0)
        //     .engine_name(app_name)
        //     .engine_version(0)
        //     .api_version(vk::make_api_version(0, 1, 0, 0));

        // let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        //     vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        // } else {
        //     vk::InstanceCreateFlags::default()
        // };

        // let create_info = vk::InstanceCreateInfo::default()
        //     .application_info(&appinfo)
        //     .enabled_layer_names(&layers_names_raw)
        //     .enabled_extension_names(&extension_names)
        //     .flags(create_flags);

        // let instance: Instance = entry
        //     .create_instance(&create_info, None)
        //     .expect("Instance creation error");
        Self { entry }
    }
}

impl RawInstance for VulkanInstance {
    fn init(&mut self, _desc: InstanceDescription) {
        unimplemented!()
    }
}
