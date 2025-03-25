use interaction::Application;

mod interaction;
mod rendering;
mod shaders;

fn main() {
    let application = Application::default();
    application.run();
}
