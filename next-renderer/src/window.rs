use ash::{extensions::khr, prelude::*, vk, Entry, Instance};
use winit::{
    dpi, error,
    event::{Event, WindowEvent},
    event_loop,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle},
    window,
};

pub struct Window {
    event_loop: event_loop::EventLoop<()>,
    native_window: window::Window,
    _width: u32,
    _height: u32,
}

impl Window {
    pub fn new(width: u32, height: u32) -> Self {
        let event_loop = event_loop::EventLoop::new().unwrap();
        let native_window = window::WindowBuilder::new()
            .with_inner_size(dpi::LogicalSize::new(width, height))
            .with_resizable(false)
            .build(&event_loop)
            .unwrap();

        Self {
            event_loop,
            native_window,
            _width: width,
            _height: height,
        }
    }

    pub unsafe fn create_surface_vk(
        &self,
        entry: &Entry,
        instance: &Instance,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::SurfaceKHR> {
        let display_handle = self.native_window.display_handle().unwrap().as_raw();
        let window_handle = self.native_window.window_handle().unwrap().as_raw();
        match (display_handle, window_handle) {
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
                let surface_desc = vk::Win32SurfaceCreateInfoKHR::builder()
                    .hinstance(window.hinstance.unwrap().get() as *mut _)
                    .hwnd(window.hwnd.get() as *mut _)
                    .build();
                let surface_fn = khr::Win32Surface::new(entry, instance);
                surface_fn.create_win32_surface(&surface_desc, allocation_callbacks)
            }

            _ => {
                panic!("Unsupported platform!")
            }
        }
    }

    pub fn run<F: Fn()>(self, update_fn: F) -> Result<(), error::EventLoopError> {
        self.event_loop
            .set_control_flow(event_loop::ControlFlow::Poll);
        #[allow(clippy::single_match)]
        self.event_loop.run(|event, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.exit(),
                WindowEvent::RedrawRequested => update_fn(),
                _ => {}
            },
            _ => {}
        })
    }
}
