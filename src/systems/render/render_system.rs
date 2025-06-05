use std::{cell::RefCell, rc::Rc, sync::Arc};

use glam::Vec4;
use vulkano::{
    descriptor_set::{layout::DescriptorSetLayout, DescriptorSet, WriteDescriptorSet},
    pipeline::Pipeline,
};
use winit::event_loop::ActiveEventLoop;

use crate::{
    core::{Camera, Particles},
    shaders,
    utils::{FpsCounter, GpuTaskExecutor, VulkanoBackend},
};

use super::{render_task::RenderTask, RenderContext};

pub struct RenderSystem {
    vulkano_backend: Option<Rc<VulkanoBackend>>,
    render_context: Option<Rc<RefCell<RenderContext>>>,
    clean_color: Vec4,
    fps_counter: FpsCounter,
}

impl RenderSystem {
    pub fn new() -> Self {
        let fps_counter = FpsCounter::new(16, 1.0);
        let clean_color = Vec4::new(0.1, 0.1, 0.1, 1.0);
        Self {
            vulkano_backend: None,
            render_context: None,
            clean_color,
            fps_counter,
        }
    }

    pub fn init(&mut self, event_loop: &ActiveEventLoop, vulkano_backend: &Rc<VulkanoBackend>) {
        self.vulkano_backend = Some(vulkano_backend.clone());
        self.render_context = Some(Rc::new(RefCell::new(RenderContext::new(
            event_loop,
            &vulkano_backend.clone(),
        ))));
    }

    pub fn request_recreate_swapchain(&mut self) {
        if let Some(render_context) = &self.render_context {
            let mut render_context = render_context.borrow_mut();
            render_context.request_recreate_swapchain();
        }
    }

    pub fn render(&mut self, camera: &Camera, particles: &Particles) {
        let vulkano_backend = self.vulkano_backend.as_ref().unwrap();
        let mut render_context = self.render_context.as_mut().unwrap().borrow_mut();
        let window = render_context.window().clone();
        render_context.cleanup_finished();

        let window_size = window.inner_size();
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        render_context.check_and_recreate_swapchain(
            self.vulkano_backend.as_ref().unwrap().memory_allocator(),
        );

        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let pipeline_layout = render_context.pipeline().layout().clone();
        let descriptor_set_layout = render_context.pipeline().layout().set_layouts()[0].clone();

        let descriptor_set = create_descriptor_set(
            vulkano_backend,
            camera,
            aspect_ratio,
            &descriptor_set_layout,
        );

        let binding = pipeline_layout.clone();

        let mut render_task = RenderTask::setup(
            &mut render_context,
            self.clean_color,
            &descriptor_set,
            &binding,
            particles,
        );

        self.vulkano_backend
            .as_ref()
            .unwrap()
            .execute(&mut render_task);

        self.fps_counter.tick();
        let fps = self.fps_counter.fps();
        window.set_title(&format!(
            "Aqua GPU -FPS: {} - Particles: {}",
            fps as u32,
            particles.count() * 9
        ));
    }

    pub fn request_redraw(&mut self) {
        if let Some(render_context) = &self.render_context {
            let render_context = render_context.borrow();
            render_context.window().request_redraw();
        }
    }
}

fn create_descriptor_set(
    vulkano_backend: &VulkanoBackend,
    camera: &Camera,
    aspect_ratio: f32,
    layout: &Arc<DescriptorSetLayout>,
) -> Arc<DescriptorSet> {
    let view_matrix = camera.view_matrix();
    let projection_matrix = camera.projection_matrix(aspect_ratio);

    let uniform_data = shaders::render::unlit::vs::Data {
        view: view_matrix.to_cols_array_2d(),
        proj: projection_matrix.to_cols_array_2d(),
    };
    let uniform_buffer = vulkano_backend
        .uniform_buffer_allocator()
        .allocate_sized()
        .unwrap();
    *uniform_buffer.write().unwrap() = uniform_data;

    DescriptorSet::new(
        vulkano_backend.descriptor_set_allocator().clone(),
        layout.clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer)],
        [],
    )
    .unwrap()
}
