#![deny(clippy::pedantic)]
mod window;

use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    io::Cursor,
    mem::{self},
};

use ash::{
    extensions::{
        ext::{DebugUtils, MetalSurface},
        khr::{
            AndroidSurface, Surface, Swapchain, WaylandSurface, Win32Surface, XcbSurface,
            XlibSurface,
        },
    },
    util::Align,
    vk::{
        self, ApplicationInfo, ExtSwapchainColorspaceFn, KhrGetPhysicalDeviceProperties2Fn,
        KhrPortabilityEnumerationFn,
    },
    Device, Entry, Instance,
};
use window::Window;
#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

struct TrianglApplication {
    window: Window,

    pub _entry: Entry,
    pub _instance: Instance,
    pub device: Device,
    pub _surface_loader: Surface,
    pub swapchain_loader: Swapchain,

    pub _physical_device: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub _queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub _surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,

    pub swapchain: vk::SwapchainKHR,
    pub _present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,

    pub _pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,
    pub _setup_command_buffer: vk::CommandBuffer,

    pub _depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub _depth_image_memory: vk::DeviceMemory,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,

    pub draw_commands_reuse_fence: vk::Fence,
    pub _setup_commands_reuse_fence: vk::Fence,
}

impl TrianglApplication {
    #[allow(clippy::vec_init_then_push)]
    #[allow(clippy::too_many_lines)]
    pub fn new(window_width: u32, window_height: u32) -> Self {
        unsafe {
            // ----------window part----------
            let window = Window::new(window_width, window_height);
            // ----------vulkan part----------

            // load vulkan library
            let entry = Entry::load().unwrap();
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
                    .any(|inst_ext| CStr::from_ptr(inst_ext.extension_name.as_ptr()) == ext)
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

                entry.create_instance(&create_info, None).unwrap()
            };

            // create vulkan surface
            let surface = window.create_surface_vk(&entry, &instance, None).unwrap();

            // select physics device
            let available_physical_devices = instance.enumerate_physical_devices().unwrap();
            let surface_loader = Surface::new(&entry, &instance);
            let (physical_device, queue_family_index) = available_physical_devices
                .iter()
                .find_map(|physical_device| {
                    instance
                        .get_physical_device_queue_family_properties(*physical_device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let is_discrete_gpu = instance
                                .get_physical_device_properties(*physical_device)
                                .device_type
                                == vk::PhysicalDeviceType::DISCRETE_GPU;
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *physical_device,
                                            u32::try_from(index).unwrap(),
                                            surface,
                                        )
                                        .unwrap();
                            if is_discrete_gpu && supports_graphic_and_surface {
                                Some((*physical_device, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable device.");
            let queue_family_index = u32::try_from(queue_family_index).unwrap();
            let device_extension_names_raw = [
                Swapchain::name().as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                KhrPortabilitySubsetFn::name().as_ptr(),
            ];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];

            let queue_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build();

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features)
                .build();

            let device: Device = instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index, 0);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0];

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap();

            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: window_width,
                    height: window_height,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .copied()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1)
                .build();

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index)
                .build();

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .build();

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();
            let setup_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image)
                        .build();
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();
            let device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);
            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(surface_resolution.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = Self::find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index)
                .build();

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory");

            let fence_create_info = vk::FenceCreateInfo::builder()
                .flags(vk::FenceCreateFlags::SIGNALED)
                .build();

            let draw_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");
            let setup_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");

            Self::record_submit_commandbuffer(
                &device,
                setup_command_buffer,
                setup_commands_reuse_fence,
                present_queue,
                &[],
                &[],
                &[],
                |device, setup_command_buffer| {
                    let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
                        .image(depth_image)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .layer_count(1)
                                .level_count(1)
                                .build(),
                        )
                        .build();

                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    );
                },
            );

            let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(depth_image)
                .format(depth_image_create_info.format)
                .view_type(vk::ImageViewType::TYPE_2D)
                .build();

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            Self {
                window,
                _entry: entry,
                _instance: instance,
                device,
                _surface_loader: surface_loader,
                swapchain_loader,
                _physical_device: physical_device,
                device_memory_properties,
                _queue_family_index: queue_family_index,
                present_queue,
                _surface: surface,
                surface_format,
                surface_resolution,
                swapchain,
                _present_images: present_images,
                present_image_views,
                _pool: pool,
                draw_command_buffer,
                _setup_command_buffer: setup_command_buffer,
                _depth_image: depth_image,
                depth_image_view,
                _depth_image_memory: depth_image_memory,
                present_complete_semaphore,
                rendering_complete_semaphore,
                draw_commands_reuse_fence,
                _setup_commands_reuse_fence: setup_commands_reuse_fence,
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn run(self) {
        let renderpass_attachments = [
            vk::AttachmentDescription {
                format: self.surface_format.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: vk::Format::D16_UNORM,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let subpass = vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build();

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&renderpass_attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies)
            .build();

        let renderpass = unsafe {
            self.device
                .create_render_pass(&renderpass_create_info, None)
                .unwrap()
        };

        let framebuffers: Vec<vk::Framebuffer> = self
            .present_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view, self.depth_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(renderpass)
                    .attachments(&framebuffer_attachments)
                    .width(self.surface_resolution.width)
                    .height(self.surface_resolution.height)
                    .layers(1)
                    .build();

                unsafe {
                    self.device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                }
            })
            .collect();

        let index_buffer_data = [0u32, 1, 2];
        let index_buffer_info = vk::BufferCreateInfo::builder()
            .size(std::mem::size_of_val(&index_buffer_data) as u64)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let index_buffer = unsafe { self.device.create_buffer(&index_buffer_info, None).unwrap() };
        let index_buffer_memory_req =
            unsafe { self.device.get_buffer_memory_requirements(index_buffer) };
        let index_buffer_memory_index = Self::find_memorytype_index(
            &index_buffer_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the index buffer.");

        let index_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: index_buffer_memory_req.size,
            memory_type_index: index_buffer_memory_index,
            ..Default::default()
        };
        let index_buffer_memory = unsafe {
            self.device
                .allocate_memory(&index_allocate_info, None)
                .unwrap()
        };
        let index_ptr = unsafe {
            self.device
                .map_memory(
                    index_buffer_memory,
                    0,
                    index_buffer_memory_req.size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap()
        };
        let mut index_slice = unsafe {
            Align::new(
                index_ptr,
                mem::align_of::<u32>() as u64,
                index_buffer_memory_req.size,
            )
        };
        index_slice.copy_from_slice(&index_buffer_data);
        unsafe { self.device.unmap_memory(index_buffer_memory) };
        unsafe {
            self.device
                .bind_buffer_memory(index_buffer, index_buffer_memory, 0)
                .unwrap();
        };

        let vertex_input_buffer_info = vk::BufferCreateInfo {
            size: 3 * std::mem::size_of::<Vertex>() as u64,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vertex_input_buffer = unsafe {
            self.device
                .create_buffer(&vertex_input_buffer_info, None)
                .unwrap()
        };

        let vertex_input_buffer_memory_req = unsafe {
            self.device
                .get_buffer_memory_requirements(vertex_input_buffer)
        };

        let vertex_input_buffer_memory_index = Self::find_memorytype_index(
            &vertex_input_buffer_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the vertex buffer.");

        let vertex_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: vertex_input_buffer_memory_req.size,
            memory_type_index: vertex_input_buffer_memory_index,
            ..Default::default()
        };

        let vertex_input_buffer_memory = unsafe {
            self.device
                .allocate_memory(&vertex_buffer_allocate_info, None)
                .unwrap()
        };

        let vertices = [
            Vertex {
                pos: [-1.0, 1.0, 0.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                pos: [1.0, 1.0, 0.0, 1.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                pos: [0.0, -1.0, 0.0, 1.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
        ];

        let vert_ptr = unsafe {
            self.device
                .map_memory(
                    vertex_input_buffer_memory,
                    0,
                    vertex_input_buffer_memory_req.size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap()
        };

        let mut vert_align = unsafe {
            Align::new(
                vert_ptr,
                mem::align_of::<Vertex>() as u64,
                vertex_input_buffer_memory_req.size,
            )
        };
        vert_align.copy_from_slice(&vertices);
        unsafe { self.device.unmap_memory(vertex_input_buffer_memory) };
        unsafe {
            self.device
                .bind_buffer_memory(vertex_input_buffer, vertex_input_buffer_memory, 0)
                .unwrap();
        };
        let mut vertex_spv_file = Cursor::new(&include_bytes!("../shaders/vert.spv")[..]);
        let mut frag_spv_file = Cursor::new(&include_bytes!("../shaders/frag.spv")[..]);

        let vertex_code = ash::util::read_spv(&mut vertex_spv_file)
            .expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::builder()
            .code(&vertex_code)
            .build();

        let frag_code = ash::util::read_spv(&mut frag_spv_file)
            .expect("Failed to read fragment shader spv file");
        let frag_shader_info = vk::ShaderModuleCreateInfo::builder()
            .code(&frag_code)
            .build();

        let vertex_shader_module = unsafe {
            self.device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };

        let fragment_shader_module = unsafe {
            self.device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error")
        };

        let layout_create_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap()
        };

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo {
                module: vertex_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                module: fragment_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: u32::try_from(mem::size_of::<Vertex>()).unwrap(),
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: u32::try_from(offset_of!(Vertex, pos)).unwrap(),
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: u32::try_from(offset_of!(Vertex, color)).unwrap(),
            },
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_input_binding_descriptions)
            .build();
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        #[allow(clippy::cast_precision_loss)]
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.surface_resolution.width as f32,
            height: self.surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [self.surface_resolution.into()];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .scissors(&scissors)
            .viewports(&viewports)
            .build();

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };
        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states)
            .build();

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_state)
            .build();

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(renderpass)
            .build();

        let graphics_pipelines = unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphic_pipeline_info],
                    None,
                )
                .expect("Unable to create graphics pipeline")
        };

        let graphic_pipeline = graphics_pipelines[0];

        self.window
            .run(|| {
                let (present_index, _) = unsafe {
                    self.swapchain_loader
                        .acquire_next_image(
                            self.swapchain,
                            std::u64::MAX,
                            self.present_complete_semaphore,
                            vk::Fence::null(),
                        )
                        .unwrap()
                };
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];

                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(renderpass)
                    .framebuffer(framebuffers[present_index as usize])
                    .render_area(self.surface_resolution.into())
                    .clear_values(&clear_values)
                    .build();

                Self::record_submit_commandbuffer(
                    &self.device,
                    self.draw_command_buffer,
                    self.draw_commands_reuse_fence,
                    self.present_queue,
                    &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                    &[self.present_complete_semaphore],
                    &[self.rendering_complete_semaphore],
                    |device, draw_command_buffer| {
                        unsafe {
                            device.cmd_begin_render_pass(
                                draw_command_buffer,
                                &render_pass_begin_info,
                                vk::SubpassContents::INLINE,
                            );
                        };
                        unsafe {
                            device.cmd_bind_pipeline(
                                draw_command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                graphic_pipeline,
                            );
                        };
                        unsafe { device.cmd_set_viewport(draw_command_buffer, 0, &viewports) };
                        unsafe { device.cmd_set_scissor(draw_command_buffer, 0, &scissors) };
                        unsafe {
                            device.cmd_bind_vertex_buffers(
                                draw_command_buffer,
                                0,
                                &[vertex_input_buffer],
                                &[0],
                            );
                        };
                        unsafe {
                            device.cmd_bind_index_buffer(
                                draw_command_buffer,
                                index_buffer,
                                0,
                                vk::IndexType::UINT32,
                            );
                        };
                        unsafe {
                            device.cmd_draw_indexed(
                                draw_command_buffer,
                                u32::try_from(index_buffer_data.len()).unwrap(),
                                1,
                                0,
                                0,
                                1,
                            );
                        };
                        // Or draw without the index buffer
                        // device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);
                        unsafe { device.cmd_end_render_pass(draw_command_buffer) };
                    },
                );
                //let mut present_info_err = mem::zeroed();
                let wait_semaphors = [self.rendering_complete_semaphore];
                let swapchains = [self.swapchain];
                let image_indices = [present_index];
                let present_info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices)
                    .build();

                unsafe {
                    self.swapchain_loader
                        .queue_present(self.present_queue, &present_info)
                        .unwrap()
                };
                println!("RedrawRequested");
            })
            .unwrap();
    }

    /// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
    /// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
    /// Make sure to create the fence in a signaled state on the first use.
    #[allow(clippy::too_many_arguments)]
    pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
        device: &Device,
        command_buffer: vk::CommandBuffer,
        command_buffer_reuse_fence: vk::Fence,
        submit_queue: vk::Queue,
        wait_mask: &[vk::PipelineStageFlags],
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        f: F,
    ) {
        unsafe {
            device
                .wait_for_fences(&[command_buffer_reuse_fence], true, u64::MAX)
                .expect("Wait for fence failed.");

            device
                .reset_fences(&[command_buffer_reuse_fence])
                .expect("Reset fences failed.");

            device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build();

            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Begin commandbuffer");
            f(device, command_buffer);
            device
                .end_command_buffer(command_buffer)
                .expect("End commandbuffer");

            let command_buffers = vec![command_buffer];

            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_mask)
                .command_buffers(&command_buffers)
                .signal_semaphores(signal_semaphores)
                .build();

            device
                .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
                .expect("queue submit failed.");
        }
    }

    pub fn find_memorytype_index(
        memory_req: &vk::MemoryRequirements,
        memory_prop: &vk::PhysicalDeviceMemoryProperties,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        #[allow(clippy::cast_possible_truncation)]
        memory_prop.memory_types[..memory_prop.memory_type_count as _]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                (1 << index) & memory_req.memory_type_bits != 0
                    && memory_type.property_flags & flags == flags
            })
            .map(|(index, _memory_type)| index as _)
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

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

fn main() {
    println!("Running begin");
    let app = TrianglApplication::new(960, 540);
    app.run();
    println!("Running end");
}
