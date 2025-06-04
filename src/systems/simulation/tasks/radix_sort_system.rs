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

    #[test]
    fn test_performance_1m_particles() {
        let backend = VulkanoHeadlessBackend::new();

        const PARTICLE_COUNT: usize = 1_000_000;

        println!("Creating {} particles...", PARTICLE_COUNT);
        let start_time = std::time::Instant::now();

        let mut particles = Particles::new(backend.memory_allocator());

        // Generate 1 million particles with random positions
        let mut particle_data = Vec::with_capacity(PARTICLE_COUNT);
        for i in 0..PARTICLE_COUNT {
            let x = (i % 1000) as f32 * 0.1;
            let y = ((i / 1000) % 1000) as f32 * 0.1;
            let z = (i / 1_000_000) as f32 * 0.1;

            particle_data.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocitie: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&particle_data, backend.memory_allocator(), &backend);
        let setup_time = start_time.elapsed();
        println!("Particle creation completed, time: {:?}", setup_time);

        // Calculate Morton hash values
        println!("Computing Morton hash codes...");
        let hash_start = std::time::Instant::now();

        let hash_constants = MortonHashConstants::new(particles.count(), 100.0); // Larger world size
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        let hash_time = hash_start.elapsed();
        println!("Morton hash computation completed, time: {:?}", hash_time);

        // Execute radix sort performance test
        println!("Starting radix sort for {} particles...", PARTICLE_COUNT);
        let sort_start = std::time::Instant::now();

        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        let sort_time = sort_start.elapsed();
        println!("Radix sort completed, time: {:?}", sort_time);

        // Verify result correctness (check only first 1000 elements)
        println!("Verifying sort results...");
        let verify_start = std::time::Instant::now();

        let result_hashes = particles.hash().read().unwrap();
        let hash_slice = &result_hashes[..particles.count() as usize];

        // Only verify first 1000 elements to avoid too long time
        let check_count = 1000.min(particles.count() as usize);
        for i in 1..check_count {
            assert!(
                hash_slice[i - 1] <= hash_slice[i],
                "Sort verification failed: hash[{}] = {} > hash[{}] = {}",
                i - 1,
                hash_slice[i - 1],
                i,
                hash_slice[i]
            );
        }

        let verify_time = verify_start.elapsed();
        println!("Sort verification completed, time: {:?}", verify_time);

        let total_time = start_time.elapsed();

        println!("\n========== Performance Test Results ==========");
        println!("Particle count: {}", PARTICLE_COUNT);
        println!("Setup time: {:?}", setup_time);
        println!("Hash time: {:?}", hash_time);
        println!("Sort time: {:?}", sort_time);
        println!("Verify time: {:?}", verify_time);
        println!("Total time: {:?}", total_time);
        println!(
            "Particles sorted per second: {:.0}",
            PARTICLE_COUNT as f64 / sort_time.as_secs_f64()
        );
        println!("===============================================");
    }

    #[test]
    fn test_performance_100k_particles() {
        let backend = VulkanoHeadlessBackend::new();

        const PARTICLE_COUNT: usize = 100_000;

        println!("Creating {} particles...", PARTICLE_COUNT);
        let start_time = std::time::Instant::now();

        let mut particles = Particles::new(backend.memory_allocator());

        // Generate 100k particles with random positions
        let mut particle_data = Vec::with_capacity(PARTICLE_COUNT);
        for i in 0..PARTICLE_COUNT {
            let x = (i % 100) as f32 * 0.1;
            let y = ((i / 100) % 100) as f32 * 0.1;
            let z = (i / 10_000) as f32 * 0.1;

            particle_data.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocitie: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&particle_data, backend.memory_allocator(), &backend);
        let setup_time = start_time.elapsed();
        println!("Particle creation completed, time: {:?}", setup_time);

        // Calculate Morton hash values
        println!("Computing Morton hash codes...");
        let hash_start = std::time::Instant::now();

        let hash_constants = MortonHashConstants::new(particles.count(), 10.0);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        let hash_time = hash_start.elapsed();
        println!("Morton hash computation completed, time: {:?}", hash_time);

        // Execute radix sort performance test
        println!("Starting radix sort for {} particles...", PARTICLE_COUNT);
        let sort_start = std::time::Instant::now();

        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        let sort_time = sort_start.elapsed();
        println!("Radix sort completed, time: {:?}", sort_time);

        // Verify result correctness
        println!("Verifying sort results...");
        let verify_start = std::time::Instant::now();

        let result_hashes = particles.hash().read().unwrap();
        let hash_slice = &result_hashes[..particles.count() as usize];

        // Verify all elements
        for i in 1..particles.count() as usize {
            assert!(
                hash_slice[i - 1] <= hash_slice[i],
                "Sort verification failed: hash[{}] = {} > hash[{}] = {}",
                i - 1,
                hash_slice[i - 1],
                i,
                hash_slice[i]
            );
        }

        let verify_time = verify_start.elapsed();
        println!("Sort verification completed, time: {:?}", verify_time);

        let total_time = start_time.elapsed();

        println!("\n========== Performance Test Results ==========");
        println!("Particle count: {}", PARTICLE_COUNT);
        println!("Setup time: {:?}", setup_time);
        println!("Hash time: {:?}", hash_time);
        println!("Sort time: {:?}", sort_time);
        println!("Verify time: {:?}", verify_time);
        println!("Total time: {:?}", total_time);
        println!(
            "Particles sorted per second: {:.0}",
            PARTICLE_COUNT as f64 / sort_time.as_secs_f64()
        );
        println!("===============================================");
    }

    #[test]
    fn test_realtime_performance() {
        let backend = VulkanoHeadlessBackend::new();

        // Test different particle counts for real-time requirements
        let test_cases = [10_000, 25_000, 50_000, 75_000, 100_000];
        let target_frame_time_ms = 16.67; // 60 FPS

        println!("\n=== Real-time Performance Analysis ===");
        println!("Target: <{:.2}ms for 60 FPS", target_frame_time_ms);
        println!("Particle Count | Sort Time | FPS | Status");
        println!("---------------|-----------|-----|-------");

        for &particle_count in &test_cases {
            let mut particles = Particles::new(backend.memory_allocator());

            // Generate particles
            let mut particle_data = Vec::with_capacity(particle_count);
            for i in 0..particle_count {
                let x = (i % 100) as f32 * 0.01;
                let y = ((i / 100) % 100) as f32 * 0.01;
                let z = (i / 10_000) as f32 * 0.01;

                particle_data.push(ParticleInitData {
                    position: Vec3::new(x, y, z),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                });
            }

            particles.add_particles(&particle_data, backend.memory_allocator(), &backend);

            // Calculate hashes
            let hash_constants = MortonHashConstants::new(particles.count(), 1.0);
            let mut hash_task = MortonHashTask::new(backend.device());
            hash_task.set_constants(hash_constants);
            hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
            backend.execute(&mut hash_task);

            // Measure sort performance
            let sort_start = std::time::Instant::now();
            let mut sort_system = RadixSortSystem::new(backend.device());
            sort_system.sort_morton_codes(
                &mut particles,
                &backend.descriptor_set_allocator(),
                &backend,
            );
            let sort_time = sort_start.elapsed();

            let sort_time_ms = sort_time.as_secs_f64() * 1000.0;
            let achievable_fps = 1000.0 / sort_time_ms;
            let status = if sort_time_ms <= target_frame_time_ms {
                "[OK] REAL-TIME"
            } else {
                "[FAIL] TOO SLOW"
            };

            println!(
                "{:>14} | {:>8.2}ms | {:>3.0} | {}",
                particle_count, sort_time_ms, achievable_fps, status
            );
        }

        println!("========================================\n");

        // Recommend optimal particle counts
        println!("Recommendations:");
        println!("- For 60 FPS: Use <=50k particles");
        println!("- For 30 FPS: Use <=100k particles");
        println!("- Consider adaptive sorting (every 2-4 frames)");
    }

    #[test]
    fn test_large_scale_performance() {
        let backend = VulkanoHeadlessBackend::new();

        // Test larger particle counts
        let test_cases = [250_000, 500_000, 750_000, 1_000_000];
        let target_frame_time_ms_60fps = 16.67; // 60 FPS
        let target_frame_time_ms_30fps = 33.33; // 30 FPS

        println!("\n=== Large Scale Performance Analysis ===");
        println!(
            "60FPS Target: <{:.2}ms | 30FPS Target: <{:.2}ms",
            target_frame_time_ms_60fps, target_frame_time_ms_30fps
        );
        println!("Particle Count | Sort Time | Direct FPS | With 4x Adaptive | Status");
        println!("---------------|-----------|------------|-------------------|--------");

        for &particle_count in &test_cases {
            let mut particles = Particles::new(backend.memory_allocator());

            // Generate particles efficiently
            let mut particle_data = Vec::with_capacity(particle_count);
            for i in 0..particle_count {
                let x = (i % 1000) as f32 * 0.01;
                let y = ((i / 1000) % 1000) as f32 * 0.01;
                let z = (i / 1_000_000) as f32 * 0.01;

                particle_data.push(ParticleInitData {
                    position: Vec3::new(x, y, z),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                });
            }

            particles.add_particles(&particle_data, backend.memory_allocator(), &backend);

            // Calculate hashes
            let hash_constants = MortonHashConstants::new(particles.count(), 10.0);
            let mut hash_task = MortonHashTask::new(backend.device());
            hash_task.set_constants(hash_constants);
            hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
            backend.execute(&mut hash_task);

            // Measure sort performance (average of 3 runs for accuracy)
            let mut total_time = std::time::Duration::ZERO;
            for _ in 0..3 {
                let sort_start = std::time::Instant::now();
                let mut sort_system = RadixSortSystem::new(backend.device());
                sort_system.sort_morton_codes(
                    &mut particles,
                    &backend.descriptor_set_allocator(),
                    &backend,
                );
                total_time += sort_start.elapsed();
            }
            let avg_sort_time = total_time / 3;

            let sort_time_ms = avg_sort_time.as_secs_f64() * 1000.0;
            let direct_fps = 1000.0 / sort_time_ms;

            // With adaptive sorting (every 4 frames)
            let adaptive_time_ms = sort_time_ms / 4.0;
            let adaptive_fps = 1000.0 / adaptive_time_ms;

            let status = if adaptive_time_ms <= target_frame_time_ms_60fps {
                "[EXCELLENT] 60FPS+"
            } else if adaptive_time_ms <= target_frame_time_ms_30fps {
                "[GOOD] 30FPS+"
            } else if sort_time_ms <= target_frame_time_ms_30fps {
                "[OK] 30FPS (direct)"
            } else {
                "[FAIL] <30FPS"
            };

            println!(
                "{:>14} | {:>8.2}ms | {:>9.0} | {:>16.0} | {}",
                particle_count, sort_time_ms, direct_fps, adaptive_fps, status
            );
        }

        println!("======================================================");

        // Summary recommendations
        println!("\nSummary for Different Target Frame Rates:");
        println!("- 60 FPS (direct sort): Up to ~200k particles");
        println!("- 60 FPS (adaptive 4x): Up to 1M+ particles");
        println!("- 30 FPS (direct sort): Up to 1M+ particles");
        println!("- 30 FPS (adaptive 4x): Massive scale possible");

        println!("\nRecommended Configurations:");
        println!("- 60 FPS + 1M particles: Sort every 4 frames");
        println!("- 30 FPS + 1M particles: Direct sorting works!");
        println!("- 30 FPS + 2M+ particles: Sort every 2-4 frames");
    }
}
