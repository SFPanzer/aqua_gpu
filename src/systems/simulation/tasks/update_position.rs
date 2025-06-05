use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::{Aabb, Particles};

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents)]
pub struct UpdatePositionConstants {
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    particle_count: u32,
    dt: f32,
}

impl UpdatePositionConstants {
    pub fn new(aabb: Aabb, particle_count: u32, dt: f32) -> Self {
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

impl ComputeGpuTaskConstants for UpdatePositionConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/update_position.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.velocity().clone()),
            WriteDescriptorSet::buffer(1, particles.position().clone()),
            WriteDescriptorSet::buffer(2, particles.predicted_position().clone()),
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type UpdatePositionTask = ComputeGpuTask<UpdatePositionConstants>;

#[cfg(test)]
mod tests {
    use crate::core::ParticlePosition;
    use crate::systems::simulation::tasks::update_position::UpdatePositionConstants;
    use crate::systems::simulation::tasks::UpdatePositionTask;
    use crate::utils::approx_eq;
    use crate::utils::VulkanoHeadlessBackend;
    use crate::{
        core::{ParticleInitData, Particles},
        utils::GpuTaskExecutor,
    };
    use glam::Vec3;
    #[test]
    fn test_update_position() {
        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocity: Vec3::new(1.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocity: Vec3::new(0.0, 1.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 1.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        particles.copy_position_to_predicted(&backend);

        {
            let mut predicted_pos = particles.predicted_position().write().unwrap();
            predicted_pos[0].position[0] = 0.1;
            predicted_pos[1].position[1] = 0.1;
            predicted_pos[2].position[2] = 0.1;
        }

        let constant = UpdatePositionConstants {
            aabb_min: [-1., -1., -1., 0.],
            aabb_max: [1., 1., 1., 0.],
            particle_count: particles.count(),
            dt: 0.1,
        };

        let mut task = UpdatePositionTask::new(backend.device());
        task.set_constants(constant);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        backend.execute(&mut task);

        let result_entries = particles.position().read().unwrap();
        let expected_entries = vec![
            ParticlePosition::new(Vec3::new(0.1, 0.0, 0.0)),
            ParticlePosition::new(Vec3::new(0.0, 0.1, 0.0)),
            ParticlePosition::new(Vec3::new(0.0, 0.0, 0.1)),
        ];
        assert_eq!(particles.count() as usize, expected_entries.len());
        for (r, e) in result_entries.iter().zip(expected_entries.iter()) {
            for (i, (a, b)) in r.position.iter().zip(e.position.iter()).enumerate() {
                assert!(
                    approx_eq(*a, *b, 1e-5),
                    "Mismatch at component {}: {} != {}",
                    i,
                    a,
                    b
                );
            }
        }

        let velocity_entries = particles.velocity().read().unwrap();
        for i in 0..3 {
            assert!(
                approx_eq(velocity_entries[i].velocity[i], 1.0, 1e-5),
                "Velocity component {} should be 1.0, but got {}",
                i,
                velocity_entries[i].velocity[i]
            );
        }
    }
}
