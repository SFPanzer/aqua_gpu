use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::systems::simulation::tasks::compute_task::ComputeGpuTask;

use super::compute_task::ComputeGpuTaskConstants;

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents)]
pub struct PrefixSumConstants {
    num_work_groups: u32,
    total_bins: u32,
}

impl PrefixSumConstants {
    #[allow(unused)]
    pub fn new(num_work_groups: u32, total_bins: u32) -> Self {
        Self {
            num_work_groups,
            total_bins,
        }
    }
}

impl ComputeGpuTaskConstants for PrefixSumConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/prefix_sum.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(
        particles: &crate::core::Particles,
    ) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.histograms().clone()),
            WriteDescriptorSet::buffer(1, particles.prefix_sums().clone()),
        ]
    }

    fn particle_count(&self) -> u32 {
        256 // Fixed workgroup size for prefix sum
    }
}

pub(crate) type PrefixSumTask = ComputeGpuTask<PrefixSumConstants>;
