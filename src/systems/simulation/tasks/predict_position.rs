use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::{Aabb, Particles};

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct PredictPositionConstants {
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    particle_count: u32,
    dt: f32,
}

impl PredictPositionConstants {
    pub fn new(particle_count: u32, dt: f32, aabb: Aabb) -> Self {
        let aabb_min = aabb.min().extend(0.).to_array();
        let aabb_max = aabb.max().extend(0.).to_array();

        Self {
            aabb_min,
            aabb_max,
            particle_count,
            dt,
        }
    }
}

impl ComputeGpuTaskConstants for PredictPositionConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/predict_position.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.position().clone()),
            WriteDescriptorSet::buffer(1, particles.velocity().clone()),
            WriteDescriptorSet::buffer(2, particles.predicted_position().clone()),
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type PredictPositionTask = ComputeGpuTask<PredictPositionConstants>;
