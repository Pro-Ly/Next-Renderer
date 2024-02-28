use next_gpu_backend::{
    instance::{Instance, InstanceDescription},
    Backend,
};
use next_renderer::window::Window;

extern crate next_gpu_backend;
extern crate next_renderer;

fn main() {
    println!("Hello, triangle!");
    let window = Window::new(800, 600);
    let _instance = Instance::new(Backend::Vulkan, &InstanceDescription::default());
    window.run(|| {}).unwrap();
}
