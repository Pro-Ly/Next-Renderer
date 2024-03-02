use std::ffi::{c_char, CStr};

use ash::{extensions::ext::DebugUtils, vk};

use crate::api::instance::{CreationFlags, Description};

pub struct Api;

impl crate::Api for Api {
    type Instance = Instance;
}

#[allow(dead_code)]
pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl crate::Instance<Api> for Instance {
    /// ## Panics
    ///
    /// Panics if the instance creation fails.
    #[must_use]
    fn new(desc: &Description) -> Self {
        let entry = ash::Entry::linked();

        log::debug!(
            "vulkan available version {}.{}.{}",
            vk::api_version_major(entry.try_enumerate_instance_version().unwrap().unwrap()),
            vk::api_version_minor(entry.try_enumerate_instance_version().unwrap().unwrap()),
            vk::api_version_patch(entry.try_enumerate_instance_version().unwrap().unwrap()),
        );

        let mut layer_names = Vec::new();
        if desc.flags().contains(CreationFlags::VALIDATION) {
            layer_names.push(unsafe {
                CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_LUNARG_standard_validation\0")
            });
        }

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut extension_names = Vec::new();

        if desc.flags().contains(CreationFlags::DEBUG) {
            extension_names.push(DebugUtils::name().as_ptr());
        }

        #[cfg(target_os = "windows")]
        {
            extension_names.push(ash::extensions::khr::Win32Surface::name().as_ptr());
        }

        #[cfg(target_os = "android")]
        {
            extension_names.push(ash::extensions::khr::AndroidSurface::name().as_ptr());
        }

        #[cfg(target_os = "linux")]
        {
            extension_names.push(ash::extensions::khr::XlibSurface::name().as_ptr());
            extension_names.push(ash::extensions::khr::XcbSurface::name().as_ptr());
            extension_names.push(ash::extensions::khr::WaylandSurface::name().as_ptr());
        }

        #[cfg(target_os = "macos")]
        {
            extension_names.push(ash::extensions::mvk::MacOSSurface::name().as_ptr());
        }

        #[cfg(target_os = "ios")]
        {
            extension_names.push(ash::extensions::mvk::IOSSurface::name().as_ptr());
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(KhrPortabilityEnumerationFn::NAME.as_ptr());
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(KhrGetPhysicalDeviceProperties2Fn::NAME.as_ptr());
        }

        let appinfo = vk::ApplicationInfo::builder()
            .application_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"next-app") })
            .application_version(0)
            .engine_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"next-gpu") })
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Instance creation error")
        };

        Self { entry, instance }
    }
}
