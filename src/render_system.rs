use std::{rc::Rc, sync::Arc};

use vulkano::{device::Device, pipeline::{layout::PipelineLayoutCreateInfo, PipelineLayout}};
use vulkano_taskgraph::graph::TaskGraph;
use winit::event_loop::ActiveEventLoop;

use crate::{render_context::RenderContext, vulkano_backend::VulkanoBackend};

pub(crate) struct RenderSystem {
    vulkan_backend: Rc<VulkanoBackend>,
    tasks: Box<dyn GpuTaskPipeline>,
    render_context: Option<RenderContext>,
}

impl RenderSystem {
    pub fn new(vulkan_backend: Rc<VulkanoBackend>, tasks: Box<dyn GpuTaskPipeline>) -> Self {
        Self {
            vulkan_backend,
            tasks,
            render_context: None,
        }
    }

    pub fn backend(&self) -> &VulkanoBackend {
        &self.vulkan_backend
    }

    pub fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let pipeline_layout = self.tasks.get_pipeline_layout(self.vulkan_backend.device());
        self.render_context = Some(RenderContext::new(event_loop, self, pipeline_layout))
    }
}

pub(crate) trait GpuTaskPipeline {
    fn get_pipeline_layout(&self, device: &Arc<Device>) -> Arc<PipelineLayout> {
        PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                ..Default::default()
            },
        ).unwrap()
    }

    fn get_task_graph(&self) -> TaskGraph<VulkanoBackend> {

    }
}
