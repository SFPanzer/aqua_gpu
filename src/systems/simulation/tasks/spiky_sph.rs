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
    spiky_kernel_factor: f32,
    grid_size: f32,
    max_neighbors: u32,
}

impl SpikySphConstants {
    pub fn new(particle_count: u32, mass: f32, smoothing_radius: f32, grid_size: f32) -> Self {
        let smoothing_radius_sq = smoothing_radius * smoothing_radius;

        // Spiky kernel factor: 15 / (π * h^6) - 与PBD约束保持一致
        let spiky_kernel_factor = 15.0 / (std::f32::consts::PI * smoothing_radius.powi(6));

        Self {
            particle_count,
            mass,
            smoothing_radius,
            smoothing_radius_sq,
            spiky_kernel_factor,
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
            WriteDescriptorSet::buffer(0, particles.predicted_position().clone()),
            WriteDescriptorSet::buffer(1, particles.density().clone()),
            WriteDescriptorSet::buffer(2, particles.index().clone()),
            WriteDescriptorSet::buffer(3, particles.cell_start().clone()),
            WriteDescriptorSet::buffer(4, particles.cell_end().clone()),
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
            BuildCellIndexConstants, BuildCellIndexTask, MortonHashConstants, MortonHashTask,
            PredictPositionConstants, PredictPositionTask, RadixSortSystem,
        };

        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // Create test particles in a simple configuration for density testing
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.1, 0.0, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.1, 0.0),
                    velocity: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        // Step 1: Predict positions
        let predict_constants = PredictPositionConstants::new(
            particles.count(),
            0.016,
            crate::core::Aabb::new(
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
            ),
        );
        let mut predict_task = PredictPositionTask::new(backend.device());
        predict_task.set_constants(predict_constants);
        predict_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut predict_task);

        // Step 2: Execute Morton hash calculation
        let hash_constants = MortonHashConstants::new(particles.count(), 0.1);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // Step 3: Execute sorting
        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        // Step 4: Build cell index table
        let cell_index_constants = BuildCellIndexConstants::new(particles.count());
        let mut cell_index_task = BuildCellIndexTask::new(backend.device());
        cell_index_task.set_constants(cell_index_constants);
        cell_index_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut cell_index_task);

        // Step 5: Execute SPH density calculation
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

    #[test]
    fn test_sph_neighbor_search_performance() {
        use crate::systems::simulation::tasks::{
            BuildCellIndexConstants, BuildCellIndexTask, MortonHashConstants, MortonHashTask,
            PredictPositionConstants, PredictPositionTask, RadixSortSystem,
        };
        use std::time::Instant;

        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // Create a larger number of particles for performance testing
        let particle_count = 10000;
        let mut test_particles = Vec::new();
        for i in 0..particle_count {
            let x = (i % 100) as f32 * 0.01; // 100x100 grid
            let y = (i / 100 % 100) as f32 * 0.01;
            let z = (i / 10000) as f32 * 0.01;
            test_particles.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocity: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&test_particles, backend.memory_allocator(), &backend);

        // Complete pipeline execution with timing
        let start_time = Instant::now();

        // Step 1: Predict positions
        let predict_constants = PredictPositionConstants::new(
            particles.count(),
            0.016,
            crate::core::Aabb::new(
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
            ),
        );
        let mut predict_task = PredictPositionTask::new(backend.device());
        predict_task.set_constants(predict_constants);
        predict_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut predict_task);

        // Step 2: Morton hash
        let hash_constants = MortonHashConstants::new(particles.count(), 0.1);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // Step 3: Sorting
        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        // Step 4: Build cell index
        let cell_index_constants = BuildCellIndexConstants::new(particles.count());
        let mut cell_index_task = BuildCellIndexTask::new(backend.device());
        cell_index_task.set_constants(cell_index_constants);
        cell_index_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut cell_index_task);

        let preprocessing_time = start_time.elapsed();

        // Step 5: SPH density calculation with timing
        let sph_start = Instant::now();
        let constants = SpikySphConstants::new(
            particles.count(),
            0.02, // mass
            0.15, // smoothing_radius (optimized)
            0.1,  // grid_size
        );

        let mut task = SpikySphTask::new(backend.device());
        task.set_constants(constants);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        backend.execute(&mut task);
        let sph_time = sph_start.elapsed();

        println!("=== SPH Cell-based Neighbor Search Performance ===");
        println!("Particle count: {}", particles.count());
        println!(
            "Preprocessing time: {:.3}ms",
            preprocessing_time.as_secs_f64() * 1000.0
        );
        println!(
            "SPH density calculation: {:.3}ms",
            sph_time.as_secs_f64() * 1000.0
        );
        println!(
            "Total time: {:.3}ms",
            (preprocessing_time + sph_time).as_secs_f64() * 1000.0
        );
        println!(
            "SPH throughput: {:.1} particles/ms",
            particles.count() as f64 / (sph_time.as_secs_f64() * 1000.0)
        );

        // Verify results
        let densities = particles.density().read().unwrap();
        let mut non_zero_count = 0;
        for i in 0..particles.count() as usize {
            if densities[i] > 0.0 {
                non_zero_count += 1;
            }
        }

        println!(
            "Particles with non-zero density: {} / {}",
            non_zero_count,
            particles.count()
        );
        assert!(
            non_zero_count > 0,
            "At least some particles should have non-zero density"
        );
        println!("==========================================");
    }

    #[test]
    fn test_cell_index_debug() {
        use crate::systems::simulation::tasks::{
            BuildCellIndexConstants, BuildCellIndexTask, MortonHashConstants, MortonHashTask,
            PredictPositionConstants, PredictPositionTask, RadixSortSystem,
        };

        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // 创建密集排列的测试粒子
        let mut test_particles = Vec::new();
        for i in 0..100 {
            let x = (i % 10) as f32 * 0.1;
            let y = (i / 10 % 10) as f32 * 0.1;
            let z = 0.0;
            test_particles.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocity: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&test_particles, backend.memory_allocator(), &backend);

        // 预测位置
        let predict_constants = PredictPositionConstants::new(
            particles.count(),
            0.016,
            crate::core::Aabb::new(
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
            ),
        );
        let mut predict_task = PredictPositionTask::new(backend.device());
        predict_task.set_constants(predict_constants);
        predict_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut predict_task);

        // Morton哈希
        let hash_constants = MortonHashConstants::new(particles.count(), 0.2);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // 排序
        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        // 构建cell索引
        let cell_index_constants = BuildCellIndexConstants::new(particles.count());
        let mut cell_index_task = BuildCellIndexTask::new(backend.device());
        cell_index_task.set_constants(cell_index_constants);
        cell_index_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut cell_index_task);

        // 检查cell索引表
        {
            let cell_start = particles.cell_start().read().unwrap();
            let cell_end = particles.cell_end().read().unwrap();

            println!("=== Cell Index Debug ===");
            let mut used_cells = 0;
            for i in 0..65536 {
                if cell_start[i] != 0xFFFFFFFF && cell_end[i] != 0xFFFFFFFF {
                    used_cells += 1;
                    if used_cells <= 10 {
                        // 只打印前10个使用的cell
                        println!(
                            "Cell {}: start={}, end={}, count={}",
                            i,
                            cell_start[i],
                            cell_end[i],
                            cell_end[i] - cell_start[i]
                        );
                    }
                }
            }
            println!("Total used cells: {}", used_cells);
        } // 释放借用

        // 现在执行SPH
        let constants = SpikySphConstants::new(
            particles.count(),
            0.02,
            0.2, // 更大的半径确保有邻居
            0.2, // 与哈希网格大小一致
        );

        let mut task = SpikySphTask::new(backend.device());
        task.set_constants(constants);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut task);

        // 检查密度结果
        let densities = particles.density().read().unwrap();
        let mut non_zero_count = 0;
        let mut total_density = 0.0;
        for i in 0..particles.count() as usize {
            if densities[i] > 0.0 {
                non_zero_count += 1;
                total_density += densities[i];
            }
        }

        println!(
            "Non-zero densities: {} / {}",
            non_zero_count,
            particles.count()
        );
        if non_zero_count > 0 {
            println!(
                "Average density: {:.6}",
                total_density / non_zero_count as f32
            );
        }

        assert!(
            non_zero_count > 0,
            "At least some particles should have non-zero density"
        );
    }

    #[test]
    fn test_performance_scale_parameters() {
        use crate::systems::simulation::{
            simulation_config::SimulationConfig,
            tasks::{
                BuildCellIndexConstants, BuildCellIndexTask, MortonHashConstants, MortonHashTask,
                PredictPositionConstants, PredictPositionTask, RadixSortSystem,
            },
        };

        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // 使用与性能测试相同的参数
        let particle_count = 1000;
        let mut test_particles = Vec::new();
        let grid_size = ((particle_count as f32).cbrt() as usize).max(10);

        for i in 0..particle_count {
            let x = (i % grid_size) as f32 * 0.02; // 与性能测试相同的间距
            let y = ((i / grid_size) % grid_size) as f32 * 0.02;
            let z = (i / (grid_size * grid_size)) as f32 * 0.02;
            test_particles.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocity: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&test_particles, backend.memory_allocator(), &backend);

        // 使用high_performance配置
        let config = SimulationConfig::high_performance();
        println!("=== Performance Test Config ===");
        println!("Grid size: {}", config.grid_size);
        println!("Smoothing radius: {}", config.sph_params.smoothing_radius);
        println!("Particle spacing: 0.02");
        println!("Particles per axis: {}", grid_size);

        // 执行完整的管道
        let predict_constants = PredictPositionConstants::new(
            particles.count(),
            0.016,
            crate::core::Aabb::new(
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
            ),
        );
        let mut predict_task = PredictPositionTask::new(backend.device());
        predict_task.set_constants(predict_constants);
        predict_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut predict_task);

        let hash_constants = MortonHashConstants::new(particles.count(), config.grid_size);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        let cell_index_constants = BuildCellIndexConstants::new(particles.count());
        let mut cell_index_task = BuildCellIndexTask::new(backend.device());
        cell_index_task.set_constants(cell_index_constants);
        cell_index_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut cell_index_task);

        let constants = SpikySphConstants::new(
            particles.count(),
            config.sph_params.particle_mass,
            config.sph_params.smoothing_radius,
            config.grid_size,
        );

        let mut task = SpikySphTask::new(backend.device());
        task.set_constants(constants);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut task);

        let densities = particles.density().read().unwrap();
        let mut non_zero_count = 0;
        let mut total_density = 0.0;
        for i in 0..particles.count() as usize {
            if densities[i] > 0.0 {
                non_zero_count += 1;
                total_density += densities[i];
            }
        }

        println!(
            "Non-zero densities: {} / {}",
            non_zero_count,
            particles.count()
        );
        if non_zero_count > 0 {
            println!(
                "Average density: {:.6}",
                total_density / non_zero_count as f32
            );
        }

        assert!(
            non_zero_count > 0,
            "Performance test parameters should produce non-zero density"
        );
    }

    #[test]
    fn test_large_scale_density() {
        use crate::systems::simulation::{
            simulation_config::SimulationConfig,
            tasks::{
                BuildCellIndexConstants, BuildCellIndexTask, MortonHashConstants, MortonHashTask,
                PredictPositionConstants, PredictPositionTask, RadixSortSystem,
            },
        };

        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // 测试50000粒子情况
        let particle_count = 50000;
        let mut test_particles = Vec::new();
        let grid_size = ((particle_count as f32).cbrt() as usize).max(10);

        for i in 0..particle_count {
            let x = (i % grid_size) as f32 * 0.02;
            let y = ((i / grid_size) % grid_size) as f32 * 0.02;
            let z = (i / (grid_size * grid_size)) as f32 * 0.02;
            test_particles.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocity: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&test_particles, backend.memory_allocator(), &backend);

        // 使用high_quality配置（50000粒子时使用的配置）
        let config = SimulationConfig::high_quality();
        println!("=== Large Scale Test Config ===");
        println!("Particle count: {}", particle_count);
        println!("Grid size: {}", config.grid_size);
        println!("Smoothing radius: {}", config.sph_params.smoothing_radius);
        println!("Particle spacing: 0.02");
        println!("Particles per axis: {}", grid_size);

        // 执行完整的管道
        let predict_constants = PredictPositionConstants::new(
            particles.count(),
            0.016,
            crate::core::Aabb::new(
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
            ),
        );
        let mut predict_task = PredictPositionTask::new(backend.device());
        predict_task.set_constants(predict_constants);
        predict_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut predict_task);

        let hash_constants = MortonHashConstants::new(particles.count(), config.grid_size);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        let cell_index_constants = BuildCellIndexConstants::new(particles.count());
        let mut cell_index_task = BuildCellIndexTask::new(backend.device());
        cell_index_task.set_constants(cell_index_constants);
        cell_index_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut cell_index_task);

        let constants = SpikySphConstants::new(
            particles.count(),
            config.sph_params.particle_mass,
            config.sph_params.smoothing_radius,
            config.grid_size,
        );

        let mut task = SpikySphTask::new(backend.device());
        task.set_constants(constants);
        task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut task);

        let densities = particles.density().read().unwrap();
        let mut non_zero_count = 0;
        let mut total_density = 0.0;
        for i in 0..particles.count() as usize {
            if densities[i] > 0.0 {
                non_zero_count += 1;
                total_density += densities[i];
            }
        }

        println!(
            "Non-zero densities: {} / {}",
            non_zero_count,
            particles.count()
        );
        if non_zero_count > 0 {
            println!(
                "Average density: {:.6}",
                total_density / non_zero_count as f32
            );
        }

        // 检查密度分布
        let mut density_ranges = [0u32; 5]; // 0, 0-1, 1-10, 10-100, >100
        for &density in densities.iter().take(particle_count) {
            if density == 0.0 {
                density_ranges[0] += 1;
            } else if density < 1.0 {
                density_ranges[1] += 1;
            } else if density < 10.0 {
                density_ranges[2] += 1;
            } else if density < 100.0 {
                density_ranges[3] += 1;
            } else {
                density_ranges[4] += 1;
            }
        }

        println!("Density distribution:");
        println!("  Zero: {}", density_ranges[0]);
        println!("  0-1: {}", density_ranges[1]);
        println!("  1-10: {}", density_ranges[2]);
        println!("  10-100: {}", density_ranges[3]);
        println!("  >100: {}", density_ranges[4]);

        assert!(
            non_zero_count > 0,
            "Large scale test should produce non-zero density"
        );
    }
}
