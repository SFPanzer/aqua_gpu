use std::rc::Rc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

use crate::{render_system::RenderSystem, vulkano_backend::VulkanoBackend};

pub(crate) struct App {
    vulkano_backend: Rc<VulkanoBackend>,

    render_system: RenderSystem,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let vulkano_backend = Rc::new(VulkanoBackend::new(event_loop));
        let render_system = RenderSystem::new(vulkano_backend.clone());

        Self {
            vulkano_backend,
            render_system,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        todo!()
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        todo!()
    }
}
