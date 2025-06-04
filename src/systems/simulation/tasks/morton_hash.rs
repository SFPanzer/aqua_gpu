use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents)]
pub struct MortonHashConstants {
    particle_count: u32,
    grid_size: f32,
}

impl MortonHashConstants {
    pub fn new(particle_count: u32, grid_size: f32) -> Self {
        Self {
            particle_count,
            grid_size,
        }
    }
}

impl ComputeGpuTaskConstants for MortonHashConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                        ty: "compute",
            path: "src/shaders/simulation/morton_hash.comp",
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
            WriteDescriptorSet::buffer(0, particles.position().clone()),
            WriteDescriptorSet::buffer(1, particles.hash().clone()),
            WriteDescriptorSet::buffer(2, particles.index().clone()),
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type MortonHashTask = ComputeGpuTask<MortonHashConstants>;

#[cfg(test)]
mod tests {

    use crate::{
        core::{ParticleInitData, Particles},
        systems::simulation::tasks::{morton_hash::MortonHashConstants, MortonHashTask},
        utils::GpuTaskExecutor,
    };

    #[test]
    fn test_morton_hash() {
        use crate::utils::VulkanoHeadlessBackend;
        use glam::Vec3;

        let backend = VulkanoHeadlessBackend::new();

        let mut particles = Particles::new(backend.memory_allocator());
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(-1.0, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, -1.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, -1.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        let constants = MortonHashConstants {
            particle_count: particles.count(),
            grid_size: 1.0,
        };
        let mut task = MortonHashTask::new(backend.device());
        task.set_constants(constants);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        backend.execute(&mut task);

        let result_entries = particles.hash().read().unwrap();
        let expected_entries = vec![
            0b0100_1001_0010_0100_1001_0010_0100_1001u32,
            0b1001_0010_0100_1001_0010_0100_1001_0010u32,
            0b0010_0100_1001_0010_0100_1001_0010_0100u32,
        ];
        assert_eq!(particles.count() as usize, expected_entries.len());
        for (r, e) in result_entries.iter().zip(expected_entries.iter()) {
            assert_eq!(r, e);
        }
    }
}
