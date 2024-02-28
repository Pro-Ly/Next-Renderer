use winit::{
    dpi, error,
    event::{Event, WindowEvent},
    event_loop,
    window::{self, WindowButtons},
};

pub struct Window {
    event_loop: event_loop::EventLoop<()>,
    #[allow(dead_code)]
    native_window: window::Window,
}

impl Window {
    /// ## Panics
    ///
    /// Attempting to create the event loop off the main thread will panic.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let event_loop = event_loop::EventLoop::new().unwrap();
        let native_window = window::WindowBuilder::new()
            .with_inner_size(dpi::LogicalSize::new(width, height))
            .with_resizable(false)
            .with_maximized(false)
            .with_enabled_buttons(WindowButtons::CLOSE | WindowButtons::MINIMIZE)
            .build(&event_loop)
            .unwrap();

        Self {
            event_loop,
            native_window,
        }
    }

    /// ## Errors
    ///
    /// Returns an error if the event loop encounters an error.
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
