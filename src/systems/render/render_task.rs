use std::{cell::RefCell, rc::Rc, sync::Arc};

use glam::Vec4;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::DescriptorSet,
    device::{Device, Queue},
    pipeline::{PipelineBindPoint, PipelineLayout},
    swapchain::{SwapchainAcquireFuture, SwapchainPresentInfo},
    sync, Validated, VulkanError,
};

use super::RenderContext;
use crate::{core::Particles, utils::GpuTask};
use vulkano::sync::GpuFuture;

pub(crate) struct RenderTask {
    render_context: Rc<RefCell<RenderContext>>,
    clean_color: Vec4,
    acquired_frame: Option<AcquiredFrame>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    pipeline_layout: Option<Arc<PipelineLayout>>,
}

impl RenderTask {
    pub fn new(render_context: Rc<RefCell<RenderContext>>, clean_color: Vec4) -> Self {
        Self {
            render_context,
            clean_color,
            acquired_frame: None,
            descriptor_set: None,
            pipeline_layout: None,
        }
    }

    pub fn update_acquire_next_image(&mut self, render_context: &mut RenderContext) {
        let (image_index, acquire_future) = match render_context.get_acquire_next_image() {
            Ok(res) => res,
            Err(_) => return,
        };
        self.acquired_frame = Some(AcquiredFrame {
            image_index,
            future: acquire_future,
        });
    }

    pub fn set_descriptor_set(
        &mut self,
        descriptor_set: Arc<DescriptorSet>,
        pipeline_layout: Arc<PipelineLayout>,
    ) {
        self.descriptor_set = Some(descriptor_set);
        self.pipeline_layout = Some(pipeline_layout);
    }
}

impl GpuTask for RenderTask {
    fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        particles: &Particles,
    ) {
        let render_context = self.render_context.borrow();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some(self.clean_color.to_array().into()),
                        Some(1.0f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        render_context.framebuffers()
                            [self.acquired_frame.as_ref().unwrap().image_index as usize]
                            .clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();
        builder
            .set_viewport(0, [render_context.viewport().clone()].into_iter().collect())
            .unwrap();
        builder
            .bind_pipeline_graphics(render_context.pipeline().clone())
            .unwrap();
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline_layout.as_ref().unwrap().clone(),
                0,
                self.descriptor_set.clone().unwrap(),
            )
            .unwrap();
        builder
            .bind_vertex_buffers(0, particles.particle_buffer().clone())
            .unwrap();
        unsafe { builder.draw(particles.particle_count(), 1, 0, 0) }.unwrap();
        builder.end_render_pass(Default::default()).unwrap();
    }

    fn submit(
        &mut self,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        queue: &Arc<Queue>,
        device: &Arc<Device>,
    ) {
        let mut render_context = self.render_context.borrow_mut();

        let acquired_frame = self.acquired_frame.take().unwrap();

        let future = render_context
            .join_future(acquired_frame.future, queue, command_buffer)
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    render_context.swapchain().clone(),
                    acquired_frame.image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                render_context.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                render_context.request_recreate_swapchain();
                render_context.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                render_context.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }
    }
}

struct AcquiredFrame {
    image_index: u32,
    future: SwapchainAcquireFuture,
}
