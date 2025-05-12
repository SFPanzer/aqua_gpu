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
    utils::{FpsCounter, VulkanoBackend},
};

use super::{render_task::RenderTask, RenderContext};

pub struct RenderSystem {
    vulkano_backend: Option<Rc<VulkanoBackend>>,
    render_context: Option<Rc<RefCell<RenderContext>>>,
    render_task: Option<RenderTask>,
    fps_counter: FpsCounter,
}

impl RenderSystem {
    pub fn new() -> Self {
        let fps_counter = FpsCounter::new(16, 1.0);
        Self {
            vulkano_backend: None,
            render_context: None,
            render_task: None,
            fps_counter,
        }
    }

    pub fn init(&mut self, event_loop: &ActiveEventLoop, vulkano_backend: &Rc<VulkanoBackend>) {
        self.vulkano_backend = Some(vulkano_backend.clone());
        self.render_context = Some(Rc::new(RefCell::new(RenderContext::new(
            event_loop,
            &vulkano_backend.clone(),
        ))));
        self.render_task = Some(RenderTask::new(
            self.render_context.as_ref().unwrap().clone(),
            Vec4::new(0.1, 0.1, 0.1, 1.0),
        ));
    }

    pub fn request_recreate_swapchain(&mut self) {
        if let Some(render_context) = &self.render_context {
            let mut render_context = render_context.borrow_mut();
            render_context.request_recreate_swapchain();
        }
    }

    pub fn descriptor_set(
        &self,
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
        let uniform_buffer = self
            .vulkano_backend
            .as_ref()
            .unwrap()
            .uniform_buffer_allocator()
            .allocate_sized()
            .unwrap();
        *uniform_buffer.write().unwrap() = uniform_data;

        DescriptorSet::new(
            self.vulkano_backend
                .as_ref()
                .unwrap()
                .descriptor_set_allocator()
                .clone(),
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )
        .unwrap()
    }

    pub fn render(&mut self, camera: &Camera, particles: &Particles) {
        let (pipeline_layout, descriptor_set_layout, aspect_ratio) = {
            let mut render_context = self.render_context.as_mut().unwrap().borrow_mut();
            render_context.cleanup_finished();

            let window_size = render_context.window().inner_size();
            if window_size.width == 0 || window_size.height == 0 {
                return;
            }

            render_context.check_and_recreate_swapchain(
                self.vulkano_backend.as_ref().unwrap().memory_allocator(),
            );

            let aspect_ratio = window_size.width as f32 / window_size.height as f32;
            let pipeline_layout = render_context.pipeline().layout().clone();
            let descriptor_set_layout = render_context.pipeline().layout().set_layouts()[0].clone();

            self.render_task
                .as_mut()
                .unwrap()
                .update_acquire_next_image(&mut render_context);

            (pipeline_layout, descriptor_set_layout, aspect_ratio)
        };

        let descriptor_set = self.descriptor_set(camera, aspect_ratio, &descriptor_set_layout);
        self.render_task
            .as_mut()
            .unwrap()
            .set_descriptor_set(descriptor_set, pipeline_layout);

        self.vulkano_backend
            .as_ref()
            .unwrap()
            .execute_gpu_task(self.render_task.as_mut().unwrap(), particles);

        self.fps_counter.tick();
        let fps = self.fps_counter.fps();
        self.update_window_title(&format!("Aqua GPU -FPS: {}", fps as u32));
    }

    pub fn request_redraw(&mut self) {
        if let Some(render_context) = &self.render_context {
            let render_context = render_context.borrow();
            render_context.window().request_redraw();
        }
    }

    fn update_window_title(&mut self, title: &str) {
        if let Some(render_context) = &self.render_context {
            let render_context = render_context.borrow();
            render_context.window().set_title(title);
        }
    }
}
