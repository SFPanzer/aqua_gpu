use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    swapchain::{Surface, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

use super::{
    render_context::RenderContext, render_pipeline::CustomRenderPipeline,
    render_vertex::RenderVertex,
};

pub struct RenderSystem {
    custom_render_pipeline: Box<dyn CustomRenderPipeline>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[RenderVertex]>,
    render_context: Option<RenderContext>,
}

impl RenderSystem {
    pub fn new(
        event_loop: &EventLoop<()>,
        custom_render_pipeline: Box<dyn CustomRenderPipeline>,
    ) -> Self {
        let instance = get_vulkan_instance(event_loop);
        let (device, queue) = get_device_and_queue(&instance, event_loop);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let vertices = [
            RenderVertex::new([-0.5, -0.25]),
            RenderVertex::new([0., 0.5]),
            RenderVertex::new([0.5, -0.25]),
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        Self {
            custom_render_pipeline,
            instance,
            device,
            queue,
            command_buffer_allocator,
            vertex_buffer,
            render_context: None,
        }
    }

    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn custom_render_pipeline(&self) -> &(dyn CustomRenderPipeline + 'static) {
        &*self.custom_render_pipeline
    }

    fn redraw(&mut self) {
        let render_context = self.render_context.as_mut().unwrap();

        let window_size = render_context.window().inner_size();
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        render_context.cleanup_finished();
        render_context.check_and_recreate_swapchain();

        let (image_index, acquire_future) = match render_context.get_acquire_next_image() {
            Ok(res) => res,
            Err(_) => return,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        render_context.framebuffers()[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [render_context.viewport().clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(render_context.pipeline().clone())
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap();

        unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

        builder
            // We leave the render pass. Note that if we had multiple subpasses we could
            // have called `next_subpass` to jump to the next subpass.
            .end_render_pass(Default::default())
            .unwrap();

        // Finish recording the command buffer by calling `end`.
        let command_buffer = builder.build().unwrap();

        let future = render_context
            .join_future(acquire_future, &self.queue, command_buffer)
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    render_context.swapchain().clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                render_context.set_gpu_futures(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                render_context.set_need_recreate_swapchain();
                render_context.set_gpu_futures(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                panic!("failed to flush future: {e}");
            }
        }
    }
}

impl ApplicationHandler for RenderSystem {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.render_context = Some(RenderContext::new(event_loop, self));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let render_context = self.render_context.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                render_context.set_need_recreate_swapchain();
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let render_context = self.render_context.as_mut().unwrap();
        render_context.window().request_redraw();
    }
}

fn get_vulkan_instance(event_loop: &EventLoop<()>) -> Arc<Instance> {
    let required_extensions = Surface::required_extensions(event_loop).unwrap();

    let library = VulkanLibrary::new().unwrap();
    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap()
}

fn get_device_and_queue(
    instance: &Arc<Instance>,
    event_loop: &EventLoop<()>,
) -> (Arc<Device>, Arc<Queue>) {
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop).unwrap()
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("no suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],

            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();

    (device, queue)
}
