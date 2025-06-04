use std::sync::Arc;
use vulkano::{descriptor_set::allocator::StandardDescriptorSetAllocator, device::Device};

use crate::{core::Particles, utils::GpuTaskExecutor};

use super::{
    prefix_sum::{PrefixSumConstants, PrefixSumTask},
    radix_sort::{RadixSortConstants, RadixSortTask},
    radix_sort_histogram::{RadixSortCountConstants, RadixSortCountTask},
};

pub struct RadixSortSystem {
    histogram_task: RadixSortCountTask,
    prefix_sum_task: PrefixSumTask,
    sort_task: RadixSortTask,
}

impl RadixSortSystem {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            histogram_task: RadixSortCountTask::new(device),
            prefix_sum_task: PrefixSumTask::new(device),
            sort_task: RadixSortTask::new(device),
        }
    }

    /// Execute complete radix sort on Morton codes
    /// Perform 4 rounds of 8-bit radix sort on 32-bit data
    pub fn sort_morton_codes(
        &mut self,
        particles: &mut Particles,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        executor: &impl GpuTaskExecutor,
    ) {
        let particle_count = particles.count();
        // Use single workgroup to avoid complex multi-workgroup coordination
        let work_group_num = 1;
        // Optimize for different data sizes
        let blocks_per_work_group = if particle_count < 25000 {
            // For small datasets, use more threads per element for better GPU utilization
            particle_count.div_ceil(256)
        } else {
            // Each thread processes 4 elements, so we need fewer work groups
            let elements_per_workgroup = 256 * 4; // 256 threads * 4 elements per thread
            particle_count.div_ceil(elements_per_workgroup)
        };

        // Execute 4 rounds of 8-bit radix sort for 32-bit Morton codes
        for pass in 0..4 {
            let shift_bits = pass * 8;

            // Step 1: Calculate histogram
            let histogram_constants = RadixSortCountConstants::new(
                particle_count,
                shift_bits,
                work_group_num,
                blocks_per_work_group,
            );
            self.histogram_task.set_constants(histogram_constants);
            self.histogram_task
                .update_descriptor_set(descriptor_set_allocator, particles);
            executor.execute(&mut self.histogram_task);

            // Step 2: Calculate prefix sum
            let prefix_sum_constants = PrefixSumConstants::new(work_group_num, 256);
            self.prefix_sum_task.set_constants(prefix_sum_constants);
            self.prefix_sum_task
                .update_descriptor_set(descriptor_set_allocator, particles);
            executor.execute(&mut self.prefix_sum_task);

            // Step 3: Reorder data
            let sort_constants = RadixSortConstants::new(
                particle_count,
                shift_bits,
                work_group_num,
                blocks_per_work_group,
            );
            self.sort_task.set_constants(sort_constants);
            self.sort_task
                .update_descriptor_set(descriptor_set_allocator, particles);
            executor.execute(&mut self.sort_task);

            // After each sort, output is in temp buffer, need to swap for next round
            particles.swap_hash_buffers();
            particles.swap_index_buffers();

            // Clear all cached descriptor sets since buffers have been swapped
            particles.descriptor_sets().clear();
        }

        // If data is in temp buffer after last iteration, need final swap
        // After 4 iterations, data should be in main buffer (0 is even)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{ParticleInitData, Particles},
        systems::simulation::tasks::{MortonHashConstants, MortonHashTask},
        utils::{GpuTaskExecutor, VulkanoHeadlessBackend},
    };
    use glam::Vec3;

    #[test]
    fn test_complete_radix_sort() {
        let backend = VulkanoHeadlessBackend::new();

        let mut particles = Particles::new(backend.memory_allocator());
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(2.0, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(1.0, 1.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 2.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 2.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(-1.0, -1.0, -1.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        // 计算莫顿哈希值
        let hash_constants = MortonHashConstants::new(particles.count(), 1.0);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // 执行完整的基数排序
        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        // 验证结果已排序
        let result_hashes = particles.hash().read().unwrap();
        let result_indices = particles.index().read().unwrap();

        println!(
            "Final result hash buffer length: {}, particle count: {}",
            result_hashes.len(),
            particles.count()
        );

        // 只检查前面粒子数量的元素
        let hash_slice = &result_hashes[..particles.count() as usize];
        let index_slice = &result_indices[..particles.count() as usize];

        println!("Hash values: {:?}", hash_slice);
        println!("Index values: {:?}", index_slice);

        // 检查莫顿码是否已排序
        for i in 1..particles.count() as usize {
            assert!(
                hash_slice[i - 1] <= hash_slice[i],
                "Morton codes not sorted: hash[{}] = {} > hash[{}] = {}",
                i - 1,
                hash_slice[i - 1],
                i,
                hash_slice[i]
            );
        }

        // Verify all original indices are present
        let mut found_indices = vec![false; particles.count() as usize];
        for &index in index_slice.iter() {
            if index < particles.count() {
                found_indices[index as usize] = true;
            }
        }
        assert!(
            found_indices.iter().all(|&found| found),
            "Not all original indices found after sorting"
        );
    }
}
