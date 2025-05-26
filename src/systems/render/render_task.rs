use std::sync::Arc;

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

pub(crate) struct RenderTask<'a> {
    render_context: &'a mut RenderContext,
    clean_color: Vec4,
    acquired_frame: AcquiredFrame,
    descriptor_set: &'a Arc<DescriptorSet>,
    pipeline_layout: &'a Arc<PipelineLayout>,
    particles: &'a Particles,
}

impl<'a> RenderTask<'a> {
    pub fn setup(
        render_context: &'a mut RenderContext,
        clean_color: Vec4,
        descriptor_set: &'a Arc<DescriptorSet>,
        pipeline_layout: &'a Arc<PipelineLayout>,
        particles: &'a Particles,
    ) -> Self {
        let (image_index, acquire_future) = render_context.get_acquire_next_image().unwrap();
        let acquired_frame = AcquiredFrame {
            image_index,
            future: Some(acquire_future),
        };

        Self {
            render_context,
            clean_color,
            acquired_frame,
            descriptor_set,
            pipeline_layout,
            particles,
        }
    }
}

impl GpuTask for RenderTask<'_> {
    fn record(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some(self.clean_color.to_array().into()),
                        Some(1.0f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        self.render_context.framebuffers()
                            [self.acquired_frame.image_index as usize]
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
            .set_viewport(
                0,
                [self.render_context.viewport().clone()]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        builder
            .bind_pipeline_graphics(self.render_context.pipeline().clone())
            .unwrap();
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline_layout.clone(),
                0,
                self.descriptor_set.clone(),
            )
            .unwrap();
        builder
            .bind_vertex_buffers(0, self.particles.position().clone())
            .unwrap();
        unsafe { builder.draw(self.particles.count(), 1, 0, 0) }.unwrap();
        builder.end_render_pass(Default::default()).unwrap();
    }

    fn submit(
        &mut self,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        queue: &Arc<Queue>,
        device: &Arc<Device>,
    ) {
        let future = self
            .render_context
            .join_future(
                self.acquired_frame.future.take().unwrap(),
                queue,
                command_buffer,
            )
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.render_context.swapchain().clone(),
                    self.acquired_frame.image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.render_context.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.render_context.request_recreate_swapchain();
                self.render_context.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                self.render_context.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }
    }
}

struct AcquiredFrame {
    image_index: u32,
    future: Option<SwapchainAcquireFuture>,
}
