use std::sync::Arc;
#[cfg(test)]
use std::time::{Duration, Instant};

use vulkano::{descriptor_set::allocator::StandardDescriptorSetAllocator, device::Device};

use crate::{core::Particles, utils::GpuTaskExecutor};

use super::{
    simulation_config::SimulationConfig,
    tasks::{
        ApplyGravityConstants, ApplyGravityTask, BuildCellIndexConstants, BuildCellIndexTask,
        MortonHashConstants, MortonHashTask, NeighborSearchConstants, NeighborSearchTask,
        PbdDensityConstraintConstants, PbdDensityConstraintTask, PredictPositionConstants,
        PredictPositionTask, RadixSortSystem, SpikySphConstants, SpikySphTask,
        UpdatePositionConstants, UpdatePositionTask,
    },
};

#[derive(Debug, Clone)]
#[cfg(test)]
pub struct SimulationStepTiming {
    pub morton_hash_time: Duration,
    pub radix_sort_time: Duration,
    pub neighbor_search_time: Duration,
    pub sph_density_time: Duration,
    pub pbd_constraint_time: Duration,
    pub gravity_time: Duration,
    pub position_update_time: Duration,
    pub total_time: Duration,
}

#[cfg(test)]
impl SimulationStepTiming {
    pub fn print_detailed(&self, particle_count: u32) {
        println!("{} 粒子仿真步骤耗时", particle_count);
        println!(
            "重力应用:       {:>8.3}ms",
            self.gravity_time.as_secs_f64() * 1000.0
        );
        println!(
            "Morton哈希计算: {:>8.3}ms",
            self.morton_hash_time.as_secs_f64() * 1000.0
        );
        println!(
            "Radix排序:      {:>8.3}ms",
            self.radix_sort_time.as_secs_f64() * 1000.0
        );
        println!(
            "邻居搜索:       {:>8.3}ms",
            self.neighbor_search_time.as_secs_f64() * 1000.0
        );
        println!(
            "SPH密度计算:    {:>8.3}ms",
            self.sph_density_time.as_secs_f64() * 1000.0
        );
        println!(
            "PBD密度约束:    {:>8.3}ms",
            self.pbd_constraint_time.as_secs_f64() * 1000.0
        );
        println!(
            "位置更新:       {:>8.3}ms",
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
        println!("\n* 步骤占比");
        println!(
            "重力应用:   {:.1}%",
            self.gravity_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "Morton哈希: {:.1}%",
            self.morton_hash_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "Radix排序:  {:.1}%",
            self.radix_sort_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "邻居搜索:   {:.1}%",
            self.neighbor_search_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "SPH密度:    {:.1}%",
            self.sph_density_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "PBD约束:    {:.1}%",
            self.pbd_constraint_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
        println!(
            "位置更新:   {:.1}%",
            self.position_update_time.as_secs_f64() * 1000.0 / total_ms * 100.0
        );
    }
}

pub(crate) struct SimulationTasks {
    pub apply_gravity: ApplyGravityTask,
    pub predict_position: PredictPositionTask,
    pub morton_hash: MortonHashTask,
    pub build_cell_index: BuildCellIndexTask,
    pub neighbor_search: NeighborSearchTask,
    pub update_position: UpdatePositionTask,
    pub spiky_sph: SpikySphTask,
    pub radix_sort: RadixSortSystem,
    pub pbd_density_constraint: PbdDensityConstraintTask,
}

impl SimulationTasks {
    pub fn new(device: &Arc<Device>) -> Self {
        let apply_gravity = ApplyGravityTask::new(device);
        let predict_position = PredictPositionTask::new(device);
        let morton_hash = MortonHashTask::new(device);
        let build_cell_index = BuildCellIndexTask::new(device);
        let neighbor_search = NeighborSearchTask::new(device);
        let update_position = UpdatePositionTask::new(device);
        let spiky_sph = SpikySphTask::new(device);
        let radix_sort = RadixSortSystem::new(device);
        let pbd_density_constraint = PbdDensityConstraintTask::new(device);

        Self {
            apply_gravity,
            predict_position,
            morton_hash,
            build_cell_index,
            neighbor_search,
            update_position,
            spiky_sph,
            radix_sort,
            pbd_density_constraint,
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

        let predict_position_constants =
            PredictPositionConstants::new(particle_count, dt, config.simulation_aabb);
        self.predict_position
            .set_constants(predict_position_constants);

        let morton_hash_constants = MortonHashConstants::new(particle_count, config.grid_size);
        self.morton_hash.set_constants(morton_hash_constants);

        let build_cell_index_constants = BuildCellIndexConstants::new(particle_count);
        self.build_cell_index
            .set_constants(build_cell_index_constants);

        // 邻居搜索常量设置
        let neighbor_search_constants = NeighborSearchConstants::new(
            particle_count,
            config.sph_params.smoothing_radius,
            config.grid_size,
            96, // max_neighbors，参考博客设置
        );
        self.neighbor_search
            .set_constants(neighbor_search_constants);

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

        // PBD密度约束常量设置
        let pbd_constraint_constants = PbdDensityConstraintConstants::new(
            particle_count,
            config.sph_params.rest_density,
            config.sph_params.smoothing_radius,
            config.sph_params.pbd_constraint_epsilon,
            config.sph_params.pbd_relaxation_factor,
            config.grid_size,
            config.simulation_aabb,
        );
        self.pbd_density_constraint
            .set_constants(pbd_constraint_constants);
    }

    pub fn update_descriptor_sets(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        self.apply_gravity
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.predict_position
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.morton_hash
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.build_cell_index
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.neighbor_search
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.update_position
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.spiky_sph
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.pbd_density_constraint
            .update_descriptor_set(descriptor_set_allocator, particles);
    }

    pub fn execute(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
        executor: &impl GpuTaskExecutor,
        config: &SimulationConfig,
    ) {
        // === 正确的PBD流体仿真流程（参考博客） ===

        // 1. 应用外力（重力）- 只更新粒子速度
        executor.execute(&mut self.apply_gravity);

        // 2. 预测积分 - 根据速度预测位置：predicted_position = position + velocity * dt
        executor.execute(&mut self.predict_position);

        // 3. 基于预测位置计算Morton哈希（为空间排序做准备）
        executor.execute(&mut self.morton_hash);

        // 4. 执行Radix排序，按Morton码对粒子排序（优化邻居搜索）
        self.radix_sort
            .sort_morton_codes(particles, descriptor_set_allocator, executor);

        // 5. 构建cell索引表，用于快速查找同一cell中的所有粒子
        executor.execute(&mut self.build_cell_index);

        // 6. 邻居搜索 - 填充contacts和contact_counts缓冲区
        executor.execute(&mut self.neighbor_search);

        // 7. 使用排序后的数据执行SPH密度计算（基于预测位置）
        executor.execute(&mut self.spiky_sph);

        // 8. PBD约束求解迭代（参考博客中的约束求解流程）
        for _ in 0..config.sph_params.pbd_iterations {
            self.pbd_density_constraint.execute_iteration(executor);
        }

        // 9. 更新最终位置和速度（应用预测位置到实际位置，并根据位置变化更新速度）
        executor.execute(&mut self.update_position);
    }

    /// Execute with detailed timing for performance analysis
    #[cfg(test)]
    pub fn execute_with_timing(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
        executor: &impl GpuTaskExecutor,
        config: &SimulationConfig,
    ) -> SimulationStepTiming {
        let total_start = Instant::now();

        // 1. 应用重力
        let gravity_start = Instant::now();
        executor.execute(&mut self.apply_gravity);
        let gravity_time = gravity_start.elapsed();

        // 2. 预测积分
        executor.execute(&mut self.predict_position);

        // 3. Morton哈希计算
        let morton_start = Instant::now();
        executor.execute(&mut self.morton_hash);
        let morton_hash_time = morton_start.elapsed();

        // 4. Radix排序
        let sort_start = Instant::now();
        self.radix_sort
            .sort_morton_codes(particles, descriptor_set_allocator, executor);
        let radix_sort_time = sort_start.elapsed();

        // 5. 构建cell索引表
        let cell_index_start = Instant::now();
        executor.execute(&mut self.build_cell_index);
        let _cell_index_time = cell_index_start.elapsed(); // 暂时不包含在输出中

        // 6. 邻居搜索
        let neighbor_start = Instant::now();
        executor.execute(&mut self.neighbor_search);
        let neighbor_search_time = neighbor_start.elapsed();

        // 7. SPH密度计算
        let sph_start = Instant::now();
        executor.execute(&mut self.spiky_sph);
        let sph_density_time = sph_start.elapsed();

        // === PBD约束求解阶段 ===

        // 8. PBD约束求解迭代
        let pbd_loop_start = Instant::now();
        for _ in 0..config.sph_params.pbd_iterations {
            self.pbd_density_constraint.execute_iteration(executor);
        }
        let pbd_constraint_time = pbd_loop_start.elapsed();

        // 9. 位置更新
        let position_start = Instant::now();
        executor.execute(&mut self.update_position);
        let position_update_time = position_start.elapsed();

        let total_time = total_start.elapsed();

        SimulationStepTiming {
            morton_hash_time,
            radix_sort_time,
            neighbor_search_time,
            sph_density_time,
            pbd_constraint_time,
            gravity_time,
            position_update_time,
            total_time,
        }
    }
}
