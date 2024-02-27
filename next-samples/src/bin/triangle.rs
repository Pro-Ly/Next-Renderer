use next_renderer::window::Window;

extern crate next_renderer;

fn main() {
    println!("Hello, triangle!");
    let window = Window::new(800, 600);
    // let instance = Instance::new(Backend::Vulkan);
    window.run(|| {}).unwrap();
}
