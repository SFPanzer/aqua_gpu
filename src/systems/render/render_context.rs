use std::sync::Arc;

use vulkano::{
    command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer},
    device::{Device, DeviceOwned, Queue},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{PolygonMode, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
    },
    sync::{self, future::JoinFuture, GpuFuture},
    Validated, VulkanError,
};
use winit::{event_loop::ActiveEventLoop, window::Window};

use crate::{core::ParticlePosition, shaders, utils::VulkanoBackend};

pub(crate) struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl RenderContext {
    pub fn new(event_loop: &ActiveEventLoop, vulkano_backend: &VulkanoBackend) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface =
            Surface::from_window(vulkano_backend.instance().clone(), window.clone()).unwrap();
        let (swapchain, images) = create_swapchain(vulkano_backend.device(), &window, surface);
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
        let render_pass = get_render_pass(vulkano_backend.device(), swapchain.image_format());
        let pipeline = get_render_pipeline(
            vulkano_backend.device(),
            &render_pass,
            &viewport,
            shaders::render::unlit::vs::load(vulkano_backend.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
            shaders::render::unlit::fs::load(vulkano_backend.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let framebuffers =
            window_size_dependent_setup(&images, &render_pass, vulkano_backend.memory_allocator());

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(vulkano_backend.device().clone()).boxed());

        Self {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain,
            previous_frame_end,
        }
    }

    pub fn window(&self) -> &Arc<Window> {
        &self.window
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

    pub fn request_recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }

    pub fn cleanup_finished(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
    }

    pub fn check_and_recreate_swapchain(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) {
        if self.recreate_swapchain {
            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: self.window.inner_size().into(),
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.swapchain = new_swapchain;

            self.framebuffers =
                window_size_dependent_setup(&new_images, &self.render_pass, memory_allocator);
            self.pipeline = get_render_pipeline(
                self.swapchain.device(),
                &self.render_pass,
                &self.viewport,
                shaders::render::unlit::vs::load(self.swapchain.device().clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap(),
                shaders::render::unlit::fs::load(self.swapchain.device().clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap(),
            );
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

    pub fn join_future<F>(
        &mut self,
        other: F,
        queue: &Arc<Queue>,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
    ) -> CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, F>>
    where
        F: GpuFuture,
    {
        self.previous_frame_end
            .take()
            .unwrap()
            .join(other)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
    }
}

fn get_render_pass(device: &Arc<Device>, format: Format) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth_stencil: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {depth_stencil},
        },
    )
    .unwrap()
}

fn get_render_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
    viewport: &Viewport,
    vertex_shader: EntryPoint,
    fragment_shader: EntryPoint,
) -> Arc<GraphicsPipeline> {
    let vertex_input_state = ParticlePosition::per_vertex()
        .definition(&vertex_shader)
        .unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vertex_shader.clone()),
        PipelineShaderStageCreateInfo::new(fragment_shader.clone()),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::PointList,
                ..Default::default()
            }),
            viewport_state: Some(ViewportState {
                viewports: [viewport.clone()].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
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
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Vec<Arc<Framebuffer>> {
    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
