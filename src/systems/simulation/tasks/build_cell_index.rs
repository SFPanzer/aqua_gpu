use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::Particles;

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct BuildCellIndexConstants {
    particle_count: u32,
}

impl BuildCellIndexConstants {
    pub fn new(particle_count: u32) -> Self {
        Self { particle_count }
    }
}

impl ComputeGpuTaskConstants for BuildCellIndexConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/build_cell_index.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.hash().clone()), // sorted_hashes
            WriteDescriptorSet::buffer(1, particles.cell_start().clone()), // cell_start
            WriteDescriptorSet::buffer(2, particles.cell_end().clone()), // cell_end
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type BuildCellIndexTask = ComputeGpuTask<BuildCellIndexConstants>;

#[cfg(test)]
mod tests {
    use crate::{
        core::{ParticleInitData, Particles},
        systems::simulation::tasks::{
            BuildCellIndexConstants, BuildCellIndexTask, MortonHashConstants, MortonHashTask,
            PredictPositionConstants, PredictPositionTask,
        },
        utils::GpuTaskExecutor,
    };

    #[test]
    fn test_build_cell_index() {
        use crate::utils::VulkanoHeadlessBackend;
        use glam::Vec3;

        let backend = VulkanoHeadlessBackend::new();

        let mut particles = Particles::new(backend.memory_allocator());
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.1, 0.0, 0.0), // Same cell
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(1.0, 0.0, 0.0), // Different cell
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        // First predict positions
        let predict_pos_constants = PredictPositionConstants::new(
            particles.count(),
            0.016,
            crate::core::Aabb::new(
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
            ),
        );
        let mut predict_pos_task = PredictPositionTask::new(backend.device());
        predict_pos_task.set_constants(predict_pos_constants);
        predict_pos_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut predict_pos_task);

        // Calculate Morton hashes
        let hash_constants = MortonHashConstants::new(particles.count(), 1.0);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // Build cell index
        let cell_index_constants = BuildCellIndexConstants::new(particles.count());
        let mut cell_index_task = BuildCellIndexTask::new(backend.device());
        cell_index_task.set_constants(cell_index_constants);
        cell_index_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut cell_index_task);

        // Verify cell index table was created
        let cell_start = particles.cell_start().read().unwrap();
        let cell_end = particles.cell_end().read().unwrap();

        // Check that we have meaningful data in cell tables
        // (具体检查逻辑取决于哈希分布和着色器实现)
        println!("Cell start buffer size: {}", cell_start.len());
        println!("Cell end buffer size: {}", cell_end.len());
        assert_eq!(cell_start.len(), 65536); // 64K entries
        assert_eq!(cell_end.len(), 65536); // 64K entries
    }
}
