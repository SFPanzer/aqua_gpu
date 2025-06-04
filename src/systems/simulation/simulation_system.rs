use std::{rc::Rc, sync::Arc, time::Instant};

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use crate::{core::Particles, utils::VulkanoBackend};

use super::{simulation_config::SimulationConfig, simulation_tasks::SimulationTasks};

pub(crate) struct SimulationSystem {
    vulkano_backend: Option<Rc<VulkanoBackend>>,
    tasks: Option<SimulationTasks>,
    config: SimulationConfig,
    last_update: Option<Instant>,
}

impl SimulationSystem {
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            vulkano_backend: None,
            tasks: None,
            config,
            last_update: None,
        }
    }

    pub fn init(&mut self, vulkano_backend: &Rc<VulkanoBackend>) {
        self.vulkano_backend = Some(vulkano_backend.clone());
        self.tasks = Some(SimulationTasks::new(vulkano_backend.device()));
    }

    pub fn update(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        let now = Instant::now();
        let dt = self.last_update.map_or(self.config.max_time_step, |last| {
            // 计算实际时间间隔，但限制在合理范围内以保证数值稳定性
            let actual_dt = now.duration_since(last).as_secs_f32();
            self.config.clamp_time_step(actual_dt)
        });
        self.last_update = Some(now);

        let tasks = self.tasks.as_mut().unwrap();
        tasks.set_constants_from_config(&self.config, particles.count(), dt);
        tasks.update_descriptor_sets(descriptor_set_allocator, particles);
        tasks.execute(
            descriptor_set_allocator,
            particles,
            self.vulkano_backend.as_ref().unwrap().as_ref(),
            &self.config,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::ParticleInitData, utils::VulkanoHeadlessBackend};
    use glam::Vec3;
    use std::time::Duration;

    #[test]
    fn test_simulation_performance_all_scales() {
        use crate::systems::simulation::simulation_tasks::SimulationStepTiming;

        println!("=== 流体仿真性能测试 - 多规模详细分析 ===");

        let test_scales = [10_000, 50_000, 100_000, 500_000, 1_000_000];
        let mut scale_results = Vec::new();

        for &particle_count in &test_scales {
            println!(
                "\n{} 测试 {} 个粒子 {}",
                "=".repeat(10),
                particle_count,
                "=".repeat(10)
            );

            let test_start = Instant::now();

            // 1. 初始化
            println!("初始化Vulkan后端和粒子系统...");
            let init_start = Instant::now();
            let headless_backend = VulkanoHeadlessBackend::new();
            let mut particles = Particles::new(headless_backend.memory_allocator());
            let init_time = init_start.elapsed();

            // 2. 生成粒子
            let particle_gen_start = Instant::now();
            let mut particle_data = Vec::with_capacity(particle_count);

            // 根据粒子数量调整生成空间
            let grid_size = ((particle_count as f32).cbrt() as usize).max(10);
            for i in 0..particle_count {
                let x = (i % grid_size) as f32 * 0.02;
                let y = ((i / grid_size) % grid_size) as f32 * 0.02;
                let z = (i / (grid_size * grid_size)) as f32 * 0.02;

                particle_data.push(ParticleInitData {
                    position: Vec3::new(x, y, z),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                });
            }

            particles.add_particles(
                &particle_data,
                headless_backend.memory_allocator(),
                &headless_backend,
            );
            let particle_gen_time = particle_gen_start.elapsed();

            // 3. 初始化仿真系统
            let sim_init_start = Instant::now();
            let config = if particle_count <= 100_000 {
                SimulationConfig::high_performance()
            } else if particle_count <= 500_000 {
                SimulationConfig::high_quality()
            } else {
                SimulationConfig::large_scale()
            };

            let mut simulation_tasks = SimulationTasks::new(headless_backend.device());
            simulation_tasks.set_constants_from_config(&config, particles.count(), 0.016);
            simulation_tasks.update_descriptor_sets(
                &headless_backend.descriptor_set_allocator(),
                &mut particles,
            );
            let sim_init_time = sim_init_start.elapsed();

            // 4. 执行详细计时测试
            println!("执行仿真步骤计时测试...");
            let frames_to_test = 5;
            let mut step_timings = Vec::with_capacity(frames_to_test);

            for frame in 0..frames_to_test {
                let timing = simulation_tasks.execute_with_timing(
                    &headless_backend.descriptor_set_allocator(),
                    &mut particles,
                    &headless_backend,
                    &config,
                );

                if frame == 0 {
                    // 只打印第一帧的详细信息
                    timing.print_detailed(particles.count());
                }

                step_timings.push(timing);
            }

            // 5. 统计分析
            let avg_timing = {
                let total_morton = step_timings
                    .iter()
                    .map(|t| t.morton_hash_time)
                    .sum::<Duration>()
                    / frames_to_test as u32;
                let total_sort = step_timings
                    .iter()
                    .map(|t| t.radix_sort_time)
                    .sum::<Duration>()
                    / frames_to_test as u32;
                let total_sph = step_timings
                    .iter()
                    .map(|t| t.sph_density_time)
                    .sum::<Duration>()
                    / frames_to_test as u32;
                let total_pbd = step_timings
                    .iter()
                    .map(|t| t.pbd_constraint_time)
                    .sum::<Duration>()
                    / frames_to_test as u32;
                let total_gravity = step_timings
                    .iter()
                    .map(|t| t.gravity_time)
                    .sum::<Duration>()
                    / frames_to_test as u32;
                let total_position = step_timings
                    .iter()
                    .map(|t| t.position_update_time)
                    .sum::<Duration>()
                    / frames_to_test as u32;
                let total_frame = step_timings.iter().map(|t| t.total_time).sum::<Duration>()
                    / frames_to_test as u32;

                SimulationStepTiming {
                    morton_hash_time: total_morton,
                    radix_sort_time: total_sort,
                    sph_density_time: total_sph,
                    pbd_constraint_time: total_pbd,
                    gravity_time: total_gravity,
                    position_update_time: total_position,
                    total_time: total_frame,
                }
            };

            let total_test_time = test_start.elapsed();
            let avg_fps = 1.0 / avg_timing.total_time.as_secs_f64();

            // 6. 性能评估
            let realtime_assessment = if avg_timing.total_time.as_millis() <= 16 {
                "✅ 支持 60+ FPS 实时仿真"
            } else if avg_timing.total_time.as_millis() <= 33 {
                "✅ 支持 30+ FPS 实时仿真"
            } else if avg_timing.total_time.as_millis() <= 100 {
                "⚠️  可接受的交互性能 (10+ FPS)"
            } else {
                "❌ 不适合实时应用 (< 10 FPS)"
            };

            println!("\n=== {} 粒子测试总结 ===", particle_count);
            println!("初始化耗时:   {:>8.3}ms", init_time.as_secs_f64() * 1000.0);
            println!(
                "粒子生成:     {:>8.3}ms",
                particle_gen_time.as_secs_f64() * 1000.0
            );
            println!(
                "系统初始化:   {:>8.3}ms",
                sim_init_time.as_secs_f64() * 1000.0
            );
            println!(
                "平均帧时间:   {:>8.3}ms",
                avg_timing.total_time.as_secs_f64() * 1000.0
            );
            println!("平均帧率:     {:>8.1} FPS", avg_fps);
            println!(
                "总测试时间:   {:>8.3}ms",
                total_test_time.as_secs_f64() * 1000.0
            );
            println!("性能评估:     {}", realtime_assessment);

            scale_results.push((particle_count, avg_timing, realtime_assessment));

            // 验证仿真结果
            let densities = particles.density().read().unwrap();
            let mut valid_density_count = 0;
            for &density in densities.iter().take(particle_count) {
                if density > 0.0 {
                    valid_density_count += 1;
                }
            }
            assert!(valid_density_count > 0, "应该至少有一些粒子具有正密度值");
        }

        // 7. 综合对比
        println!("\n\n{}", "=".repeat(60));
        println!("=== 综合性能对比 ===");
        println!("{}", "=".repeat(60));
        println!(
            "{:<12} | {:<10} | {:<8} | {:<20}",
            "粒子数量", "帧时间(ms)", "FPS", "实时性评估"
        );
        println!("{}", "-".repeat(60));

        for (count, timing, assessment) in scale_results {
            println!(
                "{:<12} | {:>10.3} | {:>8.1} | {}",
                count,
                timing.total_time.as_secs_f64() * 1000.0,
                1.0 / timing.total_time.as_secs_f64(),
                assessment
            );
        }

        println!("\n=== 性能瓶颈分析 ===");
        println!("主要耗时步骤通常为:");
        println!("1. Radix排序 - 大规模粒子时占主导");
        println!("2. SPH密度计算 - 邻域搜索复杂度");
        println!("3. Morton哈希 - 空间分割预处理");
        println!("\n建议:");
        println!("• 10万粒子以下: 适合高频率实时应用");
        println!("• 10-50万粒子: 适合游戏级别实时渲染");
        println!("• 50万以上: 适合离线仿真或低频率更新");

        println!("\n=== 测试完成 ===");
    }
}
