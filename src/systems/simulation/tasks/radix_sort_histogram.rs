use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::systems::simulation::tasks::compute_task::ComputeGpuTask;

use super::compute_task::ComputeGpuTaskConstants;

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents)]
pub struct RadixSortCountConstants {
    num_particles: u32,
    shift_bits: u32,
    num_work_groups: u32,
    num_blocks_per_work_group: u32,
}

impl RadixSortCountConstants {
    #[allow(unused)]
    pub fn new(
        num_particles: u32,
        shift_bits: u32,
        num_work_groups: u32,
        num_blocks_per_work_group: u32,
    ) -> Self {
        Self {
            num_particles,
            shift_bits,
            num_work_groups,
            num_blocks_per_work_group,
        }
    }
}

impl ComputeGpuTaskConstants for RadixSortCountConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
            ty: "compute",
            path: "src/shaders/simulation/radix_sort_histogram.comp",}
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
            WriteDescriptorSet::buffer(0, particles.hash().clone()),
            WriteDescriptorSet::buffer(1, particles.histograms().clone()),
        ]
    }

    fn particle_count(&self) -> u32 {
        self.num_particles
    }
}

pub(crate) type RadixSortCountTask = ComputeGpuTask<RadixSortCountConstants>;

#[cfg(test)]
mod tests {

    use crate::{
        core::{ParticleInitData, Particles},
        systems::simulation::tasks::{
            radix_sort_histogram::{RadixSortCountConstants, RadixSortCountTask},
            MortonHashConstants,
        },
        utils::GpuTaskExecutor,
    };

    #[test]
    fn test_radix_sort_count() {
        use crate::utils::VulkanoHeadlessBackend;
        use glam::Vec3;

        let backend = VulkanoHeadlessBackend::new();

        let mut particles = Particles::new(backend.memory_allocator());
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, -1.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, -1.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(-1.0, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, -1.0, -1.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(-1.0, 0.0, -1.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        // Calculate hash values
        let hash_constants = MortonHashConstants::new(particles.count(), 1.0);
        let mut hash_task =
            crate::systems::simulation::tasks::MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // Execute radix sort histogram
        let work_group_num = particles.count() / 256 + 1;
        let constants = RadixSortCountConstants {
            num_particles: particles.count(),
            shift_bits: 0,
            num_work_groups: work_group_num,
            num_blocks_per_work_group: 1,
        };
        let mut task = RadixSortCountTask::new(backend.device());
        task.set_constants(constants);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut task);

        // Verify results
        let result_hash_entries = particles.hash().read().unwrap();
        let result_histograms_entries = particles.histograms().read().unwrap();

        // Calculate expected bin indices (lowest 8 bits of hash values)
        let expected_bins: Vec<u32> = result_hash_entries
            .iter()
            .take(particles.count() as usize)
            .map(|&hash| hash & 0xFF)
            .collect();

        // Verify each expected bin has count 1
        for &bin in &expected_bins {
            assert_eq!(
                result_histograms_entries[bin as usize], 1,
                "Bin {} should have count 1",
                bin
            );
        }

        // Verify total number of non-zero entries equals particle count
        let non_zero_count = result_histograms_entries
            .iter()
            .filter(|&&value| value != 0)
            .count();
        assert_eq!(non_zero_count, particles.count() as usize);

        // Verify sum of all counts equals particle count
        let total_count: u32 = result_histograms_entries
            .iter()
            .take(256) // Only check first work group's 256 bins
            .sum();
        assert_eq!(total_count, particles.count());
    }
}
