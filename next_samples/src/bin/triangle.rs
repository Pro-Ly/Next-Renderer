use next_renderer::{api::instance::Description, rhi::Instance1, window::Window, Backend};

extern crate env_logger;
extern crate log;
extern crate next_renderer;

fn main() {
    println!("Hello, triangle!");
    // The `Env` lets us tweak what the environment
    // variables to read are and what the default
    // value is if they're missing
    env_logger::builder()
        .filter_level(log::LevelFilter::max())
        .init();
    let window = Window::new(800, 600);
    let _instance = Instance1::new(Backend::Vulkan, &Description::default());
    window.run(|| {}).unwrap();
}
