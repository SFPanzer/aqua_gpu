use std::rc::Rc;

use glam::{Quat, Vec3};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

use crate::{
    core::{Camera, ParticleInitData, ParticlePingPongBuffer},
    systems::{RenderSystem, SimulationConfig, SimulationSystem},
    utils::VulkanoBackend,
};

pub struct App {
    vulkano_backend: Rc<VulkanoBackend>,
    render_system: RenderSystem,
    simulation_system: SimulationSystem,
    camera: Camera,
    particles: ParticlePingPongBuffer,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let vulkano_backend = VulkanoBackend::new(event_loop);
        let render_system = RenderSystem::new();
        let simulation_system = SimulationSystem::new(SimulationConfig::default());

        let camera = Camera::new(Vec3::new(0., 0., 1.), Quat::IDENTITY, 45.0, 0.1, 100.0);
        let particles = ParticlePingPongBuffer::new(vulkano_backend.memory_allocator());

        Self {
            vulkano_backend: Rc::new(vulkano_backend),
            render_system,
            simulation_system,
            camera,
            particles,
        }
    }

    pub fn init(&mut self, event_loop: &ActiveEventLoop) {
        self.simulation_system.init(&self.vulkano_backend);
        self.render_system.init(event_loop, &self.vulkano_backend);
        self.particles.dst().add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.5, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.5, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(-0.5, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            self.vulkano_backend.memory_allocator(),
            self.vulkano_backend.as_ref(),
        );
    }

    pub fn update(&mut self) {}
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.init(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.render_system.request_recreate_swapchain();
            }
            WindowEvent::RedrawRequested => {
                self.particles.swap(self.vulkano_backend.as_ref());
                self.update();

                self.simulation_system.update(
                    self.vulkano_backend.descriptor_set_allocator(),
                    self.particles.dst(),
                );
                self.render_system
                    .render(&self.camera, self.particles.src());
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.render_system.request_redraw();
    }
}
