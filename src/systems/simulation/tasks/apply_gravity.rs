use std::sync::Arc;

use glam::Vec3;
use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::Particles;

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct ApplyGravityConstants {
    gravity: [f32; 4],
    particle_count: u32,
    dt: f32,
}

impl ApplyGravityConstants {
    pub fn new(particle_count: u32, dt: f32, gravity: Vec3) -> Self {
        Self {
            particle_count,
            dt,
            gravity: gravity.extend(0.0).into(),
        }
    }
}

impl ComputeGpuTaskConstants for ApplyGravityConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/apply_gravity.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [WriteDescriptorSet::buffer(0, particles.velocity().clone())]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type ApplyGravityTask = ComputeGpuTask<ApplyGravityConstants>;

#[cfg(test)]
mod tests {
    use crate::utils::approx_eq;
    use crate::utils::VulkanoHeadlessBackend;
    use crate::{
        core::{ParticleInitData, ParticleVelocity, Particles},
        systems::simulation::tasks::{apply_gravity::ApplyGravityConstants, ApplyGravityTask},
        utils::GpuTaskExecutor,
    };
    use glam::Vec3;

    #[test]
    fn test_apply_gravity() {
        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocitie: Vec3::new(1.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 1.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 1.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        let constant = ApplyGravityConstants {
            particle_count: particles.count(),
            dt: 0.1,
            gravity: [1.0, 2.0, 3.0, 0.0],
        };

        let mut task = ApplyGravityTask::new(backend.device());
        task.set_constants(constant);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        backend.execute(&mut task);

        let result_entries = particles.velocity().read().unwrap();
        let expected_entries = vec![
            ParticleVelocity::new(Vec3::new(1.1, 0.2, 0.3)),
            ParticleVelocity::new(Vec3::new(0.1, 1.2, 0.3)),
            ParticleVelocity::new(Vec3::new(0.1, 0.2, 1.3)),
        ];
        assert_eq!(particles.count() as usize, expected_entries.len());
        for (r, e) in result_entries.iter().zip(expected_entries.iter()) {
            for (i, (a, b)) in r.velocity.iter().zip(e.velocity.iter()).enumerate() {
                assert!(
                    approx_eq(*a, *b, 1e-5),
                    "Mismatch at component {}: {} != {}",
                    i,
                    a,
                    b
                );
            }
        }
    }
}
