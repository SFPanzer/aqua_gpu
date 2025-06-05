use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::systems::simulation::tasks::compute_task::ComputeGpuTask;

use super::compute_task::ComputeGpuTaskConstants;

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents)]
pub struct RadixSortConstants {
    num_particles: u32,
    shift_bits: u32,
    num_work_groups: u32,
    num_blocks_per_work_group: u32,
}

impl RadixSortConstants {
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

impl ComputeGpuTaskConstants for RadixSortConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/radix_sort.comp",
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
            WriteDescriptorSet::buffer(0, particles.hash().clone()), // hashes_in
            WriteDescriptorSet::buffer(1, particles.hash_temp().clone()), // hashes_out
            WriteDescriptorSet::buffer(2, particles.index().clone()), // indices_in
            WriteDescriptorSet::buffer(3, particles.index_temp().clone()), // indices_out
            WriteDescriptorSet::buffer(4, particles.prefix_sums().clone()), // prefix_sums
        ]
    }

    fn particle_count(&self) -> u32 {
        self.num_particles
    }
}

pub(crate) type RadixSortTask = ComputeGpuTask<RadixSortConstants>;

#[cfg(test)]
mod tests {
    use crate::{
        core::{ParticleInitData, Particles},
        systems::simulation::tasks::{
            radix_sort::{RadixSortConstants, RadixSortTask},
            radix_sort_histogram::{RadixSortCountConstants, RadixSortCountTask},
            MortonHashConstants, MortonHashTask,
        },
        utils::GpuTaskExecutor,
    };

    #[test]
    fn test_radix_sort() {
        use crate::utils::VulkanoHeadlessBackend;
        use glam::Vec3;

        let backend = VulkanoHeadlessBackend::new();

        let mut particles = Particles::new(backend.memory_allocator());
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 1.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 1.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(-1.0, 0.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, -1.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        // Step 1: Calculate Morton hash values
        let hash_constants = MortonHashConstants::new(particles.count(), 1.0);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // Step 2: Generate histograms for radix sort
        let work_group_num = (particles.count() + 255) / 256;
        let histogram_constants = RadixSortCountConstants::new(
            particles.count(),
            0, // Start with least significant 8 bits
            work_group_num,
            1,
        );
        let mut histogram_task = RadixSortCountTask::new(backend.device());
        histogram_task.set_constants(histogram_constants);
        histogram_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut histogram_task);

        // Step 3: Calculate prefix sums
        use crate::systems::simulation::tasks::prefix_sum::{PrefixSumConstants, PrefixSumTask};
        let prefix_sum_constants = PrefixSumConstants::new(work_group_num, 256);
        let mut prefix_sum_task = PrefixSumTask::new(backend.device());
        prefix_sum_task.set_constants(prefix_sum_constants);
        prefix_sum_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut prefix_sum_task);

        // Step 4: Execute radix sort reordering
        let sort_constants = RadixSortConstants::new(
            particles.count(),
            0, // Start with least significant 8 bits
            work_group_num,
            1,
        );
        let mut sort_task = RadixSortTask::new(backend.device());
        sort_task.set_constants(sort_constants);
        sort_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut sort_task);

        // Verify that hashes are sorted (at least partially for the first 8 bits)
        let result_hashes = particles.hash_temp().read().unwrap();
        let _result_indices = particles.index_temp().read().unwrap();

        // Check that the sort produced valid results
        println!(
            "Result hash buffer length: {}, particle count: {}",
            result_hashes.len(),
            particles.count()
        );
        // 缓冲区大小是固定的，所以只检查前面粒子数量的元素
        let hash_slice = &result_hashes[..particles.count() as usize];

        // Verify that the first 8 bits are sorted
        for i in 1..particles.count() as usize {
            let prev_radix = hash_slice[i - 1] & 0xFF;
            let curr_radix = hash_slice[i] & 0xFF;
            assert!(
                prev_radix <= curr_radix,
                "Radix sort failed: hash[{}] & 0xFF = {} > hash[{}] & 0xFF = {}",
                i - 1,
                prev_radix,
                i,
                curr_radix
            );
        }
    }
}
