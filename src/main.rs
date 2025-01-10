use application::Application;

mod application;
mod vulkan_context;

fn main() {
    let mut app = Application::initialize("Aqua GPU Simulator", 800, 600);
    app.main_loop();
}
