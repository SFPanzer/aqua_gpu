use winit::event_loop::EventLoop;

use crate::rendering::{CustomRenderPipeline, DefaultRenderPipeline, RenderSystem};

pub struct Application {
    event_loop: EventLoop<()>,
    render_system: RenderSystem,
}

impl Application {
    #[allow(unused)]
    pub fn new<T>() -> Self
    where
        T: CustomRenderPipeline + Default + 'static,
    {
        let event_loop = EventLoop::new().unwrap();

        let render_system = RenderSystem::new(&event_loop, Box::new(T::default()));

        Self {
            event_loop,
            render_system,
        }
    }

    pub fn run(mut self) {
        let _ = self.event_loop.run_app(&mut self.render_system);
    }
}

impl Default for Application {
    fn default() -> Self {
        let event_loop = EventLoop::new().unwrap();
        let render_system =
            RenderSystem::new(&event_loop, Box::new(DefaultRenderPipeline::default()));

        Self {
            event_loop,
            render_system,
        }
    }
}
