use std::sync::Arc;

use vulkano::{
    command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    image::{view::ImageView, Image, ImageUsage},
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
    },
    sync::{self, future::JoinFuture, GpuFuture},
    Validated, VulkanError,
};
use winit::{event_loop::ActiveEventLoop, window::Window};

use super::RenderSystem;

pub struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    gpu_future: Option<Box<dyn GpuFuture>>,
}

impl RenderContext {
    pub fn new(event_loop: &ActiveEventLoop, render_system: &mut RenderSystem) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(render_system.instance(), window.clone()).unwrap();

        let (swapchain, images) = create_swapchain(&render_system.device(), &window, surface);

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let render_pass = render_system
            .custom_render_pipeline()
            .get_render_pass(&render_system.device(), swapchain.image_format());
        let pipeline = render_system
            .custom_render_pipeline()
            .get_render_pipeline(&render_system.device(), &render_pass);

        let framebuffers = window_size_dependent_setup(&images, &render_pass);

        let recreate_swapchain = false;
        let gpu_future = Some(sync::now(render_system.device()).boxed());

        Self {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain,
            gpu_future,
        }
    }

    pub fn window(&mut self) -> &Arc<Window> {
        &self.window
    }

    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }

    pub fn framebuffers(&self) -> &Vec<Arc<Framebuffer>> {
        &self.framebuffers
    }

    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    pub fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        &self.pipeline
    }

    pub fn join_future<F>(
        &mut self,
        other: F,
        queue: &Arc<Queue>,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
    ) -> CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, F>>
    where
        F: GpuFuture,
    {
        self.gpu_future
            .take()
            .unwrap()
            .join(other)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
    }

    pub fn cleanup_finished(&mut self) {
        self.gpu_future.as_mut().unwrap().cleanup_finished();
    }

    pub fn set_gpu_futures(&mut self, new: Box<dyn GpuFuture>) {
        self.gpu_future = Some(new)
    }

    pub fn set_need_recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn check_and_recreate_swapchain(&mut self) {
        if self.recreate_swapchain {
            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: self.window.inner_size().into(),
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.swapchain = new_swapchain;

            self.framebuffers = window_size_dependent_setup(&new_images, &self.render_pass);
            self.viewport.extent = self.window.inner_size().into();
            self.recreate_swapchain = false;
        }
    }

    pub fn get_acquire_next_image(&mut self) -> Result<(u32, SwapchainAcquireFuture), ()> {
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return Err(());
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        Ok((image_index, acquire_future))
    }
}

fn create_swapchain(
    device: &Arc<Device>,
    window: &Arc<Window>,
    surface: Arc<Surface>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    let image_format = device
        .physical_device()
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;

    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count.max(2),
            image_format,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),

            ..Default::default()
        },
    )
    .unwrap()
}

pub fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
