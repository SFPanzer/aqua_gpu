use std::sync::Arc;
#[cfg(test)]
use std::time::{Duration, Instant};

use vulkano::{descriptor_set::allocator::StandardDescriptorSetAllocator, device::Device};

use crate::{core::Particles, utils::GpuTaskExecutor};

use super::{
    simulation_config::SimulationConfig,
    tasks::{
        ApplyGravityConstants, ApplyGravityTask, MortonHashConstants, MortonHashTask,
        RadixSortSystem, SpikySphConstants, SpikySphTask, UpdatePositionConstants,
        UpdatePositionTask,
    },
};

#[derive(Debug, Clone)]
#[cfg(test)]
pub struct SimulationStepTiming {
    pub morton_hash_time: Duration,
    pub radix_sort_time: Duration,
    pub sph_density_time: Duration,
    pub gravity_time: Duration,
    pub position_update_time: Duration,
    pub total_time: Duration,
}

#[cfg(test)]
impl SimulationStepTiming {
    pub fn print_detailed(&self, particle_count: u32) {
        println!("=== 仿真步骤详细耗时 ({} 粒子) ===", particle_count);
        println!(
            "1. Morton哈希计算: {:>8.3}ms",
            self.morton_hash_time.as_secs_f64() * 1000.0
        );
        println!(
            "2. Radix排序:      {:>8.3}ms",
            self.radix_sort_time.as_secs_f64() * 1000.0
        );
        println!(
            "3. SPH密度计算:    {:>8.3}ms",
            self.sph_density_time.as_secs_f64() * 1000.0
        );
        println!(
            "4. 重力应用:       {:>8.3}ms",
            self.gravity_time.as_secs_f64() * 1000.0
        );
        println!(
            "5. 位置更新:       {:>8.3}ms",
            self.position_update_time.as_secs_f64() * 1000.0
        );
        println!(
            "总计:              {:>8.3}ms",
            self.total_time.as_secs_f64() * 1000.0
        );
        println!(
            "有效帧率:          {:>8.1} FPS",
            1.0 / self.total_time.as_secs_f64()
        );

        // 各步骤占比
        let total_ms = self.total_time.as_secs_f64() * 1000.0;
        println!("\n=== 步骤占比 ===");
        println!(
            "Morton哈希: {:.1}%",
            self.morton_hash_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "Radix排序:  {:.1}%",
            self.radix_sort_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "SPH密度:    {:.1}%",
            self.sph_density_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "重力应用:   {:.1}%",
            self.gravity_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "位置更新:   {:.1}%",
            self.position_update_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!("==============================");
    }
}

pub(crate) struct SimulationTasks {
    pub apply_gravity: ApplyGravityTask,
    pub morton_hash: MortonHashTask,
    pub update_position: UpdatePositionTask,
    pub spiky_sph: SpikySphTask,
    pub radix_sort: RadixSortSystem,
}

impl SimulationTasks {
    pub fn new(device: &Arc<Device>) -> Self {
        let apply_gravity = ApplyGravityTask::new(device);
        let morton_hash = MortonHashTask::new(device);
        let update_position = UpdatePositionTask::new(device);
        let spiky_sph = SpikySphTask::new(device);
        let radix_sort = RadixSortSystem::new(device);

        Self {
            apply_gravity,
            morton_hash,
            update_position,
            spiky_sph,
            radix_sort,
        }
    }

    /// Set all constants using SimulationConfig
    ///
    /// * `config` - Simulation configuration parameters
    /// * `particle_count` - Number of particles
    /// * `dt` - Dynamically calculated time step (calculated from actual frame time, already limited by config.clamp_time_step())
    pub fn set_constants_from_config(
        &mut self,
        config: &SimulationConfig,
        particle_count: u32,
        dt: f32,
    ) {
        let apply_gravity_constants =
            ApplyGravityConstants::new(particle_count, dt, config.gravity);
        self.apply_gravity.set_constants(apply_gravity_constants);

        let morton_hash_constants = MortonHashConstants::new(particle_count, config.grid_size);
        self.morton_hash.set_constants(morton_hash_constants);

        let update_position_constants =
            UpdatePositionConstants::new(config.simulation_aabb, particle_count, dt);
        self.update_position
            .set_constants(update_position_constants);

        // SPH density calculation constants setup (for PBD) - using parameters from configuration
        let spiky_sph_constants = SpikySphConstants::new(
            particle_count,
            config.sph_params.particle_mass,
            config.sph_params.smoothing_radius,
            config.grid_size,
        );
        self.spiky_sph.set_constants(spiky_sph_constants);
    }

    pub fn update_descriptor_sets(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        self.apply_gravity
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.morton_hash
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.update_position
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.spiky_sph
            .update_descriptor_set(descriptor_set_allocator, particles);
    }

    pub fn execute(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
        executor: &impl GpuTaskExecutor,
    ) {
        // 1. Execute hash calculation (prepare for sorting)
        executor.execute(&mut self.morton_hash);

        // 2. Execute radix sort, sort particles by Morton code
        self.radix_sort
            .sort_morton_codes(particles, descriptor_set_allocator, executor);

        // 3. Execute SPH density calculation using sorted data
        executor.execute(&mut self.spiky_sph);

        // 4. Execute gravity
        executor.execute(&mut self.apply_gravity);

        // 5. Update positions
        executor.execute(&mut self.update_position);
    }

    /// Execute with detailed timing for performance analysis
    #[cfg(test)]
    pub fn execute_with_timing(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
        executor: &impl GpuTaskExecutor,
    ) -> SimulationStepTiming {
        let total_start = Instant::now();

        // 1. Execute hash calculation (prepare for sorting)
        let morton_start = Instant::now();
        executor.execute(&mut self.morton_hash);
        let morton_hash_time = morton_start.elapsed();

        // 2. Execute radix sort, sort particles by Morton code
        let sort_start = Instant::now();
        self.radix_sort
            .sort_morton_codes(particles, descriptor_set_allocator, executor);
        let radix_sort_time = sort_start.elapsed();

        // 3. Execute SPH density calculation using sorted data
        let sph_start = Instant::now();
        executor.execute(&mut self.spiky_sph);
        let sph_density_time = sph_start.elapsed();

        // 4. Execute gravity
        let gravity_start = Instant::now();
        executor.execute(&mut self.apply_gravity);
        let gravity_time = gravity_start.elapsed();

        // 5. Update positions
        let position_start = Instant::now();
        executor.execute(&mut self.update_position);
        let position_update_time = position_start.elapsed();

        let total_time = total_start.elapsed();

        SimulationStepTiming {
            morton_hash_time,
            radix_sort_time,
            sph_density_time,
            gravity_time,
            position_update_time,
            total_time,
        }
    }
}
