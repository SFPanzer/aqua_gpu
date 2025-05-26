use std::sync::Arc;

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::{self, Queue},
};

pub(crate) trait GpuTask {
    fn record(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>);
    fn submit(
        &mut self,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        queue: &Arc<Queue>,
        device: &Arc<device::Device>,
    );
}

pub(crate) trait GpuTaskExecutor {
    fn execute(&self, task: &mut dyn GpuTask);
}
