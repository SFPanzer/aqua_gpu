use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::Particles;

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

/// SPH density calculation task specifically for PBD fluid simulation
/// Only calculates particle density, not pressure or viscosity forces
/// Density results will be used for PBD constraint solving
///
/// PBD fluid SPH density calculation constants
#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct SpikySphConstants {
    particle_count: u32,
    mass: f32,
    smoothing_radius: f32,
    smoothing_radius_sq: f32,
    poly6_kernel_factor: f32,
    grid_size: f32,
    max_neighbors: u32,
}

impl SpikySphConstants {
    pub fn new(particle_count: u32, mass: f32, smoothing_radius: f32, grid_size: f32) -> Self {
        let smoothing_radius_sq = smoothing_radius * smoothing_radius;

        // Poly6 kernel factor: 315 / (64 * π * h^9)
        let poly6_kernel_factor = 315.0 / (64.0 * std::f32::consts::PI * smoothing_radius.powi(9));

        Self {
            particle_count,
            mass,
            smoothing_radius,
            smoothing_radius_sq,
            poly6_kernel_factor,
            grid_size,
            max_neighbors: 64, // Limit to 64 neighborhood particles
        }
    }
}

impl ComputeGpuTaskConstants for SpikySphConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/spiky_sph.comp",
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
            WriteDescriptorSet::buffer(1, particles.density().clone()),
            WriteDescriptorSet::buffer(2, particles.index().clone()), // Sorted indices
            WriteDescriptorSet::buffer(3, particles.hash().clone()),  // Morton hash values
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type SpikySphTask = ComputeGpuTask<SpikySphConstants>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{ParticleInitData, Particles},
        utils::{GpuTaskExecutor, VulkanoHeadlessBackend},
    };
    use glam::Vec3;

    #[test]
    fn test_spiky_sph_density_calculation() {
        use crate::systems::simulation::tasks::{
            MortonHashConstants, MortonHashTask, RadixSortSystem,
        };

        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // Create test particles in a simple configuration for density testing
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.1, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.1, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        // First execute Morton hash calculation
        let hash_constants = MortonHashConstants::new(particles.count(), 0.1);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // Then execute sorting
        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        // Now execute SPH density calculation
        let constants = SpikySphConstants::new(
            particles.count(),
            0.02, // mass: 0.02 kg per particle
            0.2,  // smoothing_radius: 20cm
            0.1,  // grid_size: 10cm
        );

        let mut task = SpikySphTask::new(backend.device());
        task.set_constants(constants);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        backend.execute(&mut task);

        // Verify that the density calculation ran without errors
        // The density values will be used by PBD constraint solver
        let densities = particles.density().read().unwrap();
        assert!(densities.len() >= particles.count() as usize);

        let velocities = particles.velocity().read().unwrap();
        assert!(velocities.len() >= particles.count() as usize);

        // Check if the first few particles have reasonable density values (should be greater than 0)
        // These density values will be used by PBD constraint solver for position correction
        for i in 0..particles.count() as usize {
            assert!(
                densities[i] > 0.0,
                "Density should be positive for particle {}",
                i
            );
        }
    }
}
