use std::sync::Arc;

use vulkano::{
    descriptor_set::layout::DescriptorSetLayout, device::Device, format::{Format, NumericFormat}, image::ImageUsage, pipeline::{graphics::viewport::Viewport, layout::PipelineLayoutCreateInfo, PipelineLayout}, swapchain::{ColorSpace, Surface, Swapchain, SwapchainCreateInfo}
};
use vulkano_taskgraph::Id;
use winit::{event_loop::ActiveEventLoop, window::Window};

use crate::render_system::RenderSystem;

pub struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    pipeline_layout: Arc<PipelineLayout>,
    viewport: Viewport,
    recreate_swapchain: bool,
}

impl RenderContext {
    pub fn new(event_loop: &ActiveEventLoop, render_system: &mut RenderSystem, pipeline_layout: Arc<PipelineLayout>) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface =
            Surface::from_window(render_system.backend().instance().clone(), window.clone())
                .unwrap();

        let (swapchain_format, create_info) =
            get_swapchain_create_info(render_system.backend().device(), &surface, &window);
        let swapchain_id = render_system
            .backend()
            .create_swapchain(&surface, create_info);

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let recreate_swapchain = false;

        Self {
            window,
            swapchain_id,
            pipeline_layout,
            viewport,
            recreate_swapchain,
        }
    }
}

fn get_swapchain_create_info(
    device: &Device,
    surface: &Surface,
    window: &Window,
) -> (Format, SwapchainCreateInfo) {
    let swapchain_format;
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(surface, Default::default())
        .unwrap();
    (swapchain_format, _) = device
        .physical_device()
        .surface_formats(surface, Default::default())
        .unwrap()
        .into_iter()
        .find(|&(format, color_space)| {
            format.numeric_format_color() == Some(NumericFormat::SRGB)
                && color_space == ColorSpace::SrgbNonLinear
        })
        .unwrap();

    let create_info = SwapchainCreateInfo {
        min_image_count: surface_capabilities.min_image_count.max(3),
        image_format: swapchain_format,
        image_extent: window.inner_size().into(),
        image_usage: ImageUsage::COLOR_ATTACHMENT,
        composite_alpha: surface_capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap(),
        ..Default::default()
    };

    (swapchain_format, create_info)
}
