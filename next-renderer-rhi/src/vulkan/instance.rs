use std::{
    borrow::Cow,
    ffi::{CStr, CString},
};

use ash::{
    extensions::{
        ext::{DebugUtils, MetalSurface},
        khr::{AndroidSurface, Surface, WaylandSurface, Win32Surface, XcbSurface, XlibSurface},
    },
    vk::{
        self, ApplicationInfo, ExtSwapchainColorspaceFn, KhrGetPhysicalDeviceProperties2Fn,
        KhrPortabilityEnumerationFn,
    },
    Entry,
};

#[allow(dead_code)]
pub struct VulkanInstance {
    instance: ash::Instance,
    entry: ash::Entry,
}

impl VulkanInstance {
    #[allow(clippy::vec_init_then_push)]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let entry = unsafe { Entry::load().unwrap() };
        let version = entry.try_enumerate_instance_version();
        let api_version = version.unwrap().unwrap();
        println!(
            "Vulkan api_version: {}.{}.{}",
            vk::api_version_major(api_version),
            vk::api_version_minor(api_version),
            vk::api_version_patch(api_version)
        );
        // create vulkan app
        let app_name = CString::new("Triangle").unwrap();
        let app_info = ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(1)
            .engine_name(CStr::from_bytes_with_nul(b"next-renderer\0").unwrap())
            .engine_version(2)
            .api_version(vk::API_VERSION_1_0);

        // select vulkan extensions
        let mut extensions: Vec<&'static CStr> = Vec::new();
        extensions.push(DebugUtils::name());
        extensions.push(Surface::name());
        // Platform-specific WSI extensions
        if cfg!(all(
            unix,
            not(target_os = "android"),
            not(target_os = "macos")
        )) {
            // VK_KHR_xlib_surface
            extensions.push(XlibSurface::name());
            // VK_KHR_xcb_surface
            extensions.push(XcbSurface::name());
            // VK_KHR_wayland_surface
            extensions.push(WaylandSurface::name());
        }
        if cfg!(target_os = "android") {
            // VK_KHR_android_surface
            extensions.push(AndroidSurface::name());
        }
        if cfg!(target_os = "windows") {
            // VK_KHR_win32_surface
            extensions.push(Win32Surface::name());
        }
        if cfg!(target_os = "macos") {
            // VK_EXT_metal_surface
            extensions.push(MetalSurface::name());
            extensions.push(KhrPortabilityEnumerationFn::name());
        }
        // VK_EXT_swapchain_colorspace
        // Provid wide color gamut
        extensions.push(ExtSwapchainColorspaceFn::name());

        // VK_KHR_get_physical_device_properties2
        // Even though the extension was promoted to Vulkan 1.1, we still require the extension
        // so that we don't have to conditionally use the functions provided by the 1.1 instance
        extensions.push(KhrGetPhysicalDeviceProperties2Fn::name());

        let instance_extensions = entry.enumerate_instance_extension_properties(None).unwrap();
        // Only keep available extensions.
        extensions.retain(|&ext| {
            if instance_extensions
                .iter()
                .any(|inst_ext| unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) } == ext)
            {
                true
            } else {
                panic!("Unable to find extension: {}", ext.to_string_lossy());
            }
        });

        // select vulkan layers
        let mut layers: Vec<&'static CStr> = Vec::new();
        layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());

        // setup debug messenger
        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                //     | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                //     |
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_callback))
            .build();

        // create vulkan instance
        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };
        let instance = {
            let str_pointers = layers
                .iter()
                .chain(extensions.iter())
                .map(|&s: &&'static _| {
                    // Safe because `layers` and `extensions` entries have static lifetime.
                    s.as_ptr()
                })
                .collect::<Vec<_>>();

            let create_info = vk::InstanceCreateInfo::builder()
                .flags(create_flags)
                .application_info(&app_info)
                .enabled_layer_names(&str_pointers[..layers.len()])
                .enabled_extension_names(&str_pointers[layers.len()..]);

            let create_info = create_info.push_next(&mut debug_info);

            unsafe { entry.create_instance(&create_info, None).unwrap() }
        };

        Self { instance, entry }
    }

    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    #[inline]
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = *p_callback_data;
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        println!(
            "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );

        vk::FALSE
    }
}
