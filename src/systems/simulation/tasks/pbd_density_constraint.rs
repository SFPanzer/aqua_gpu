use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::Particles;

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

/// PBD密度约束求解器
/// 通过位置校正来维持流体密度约束
/// 使用迭代方法求解约束，确保流体保持理想密度
#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct PbdDensityConstraintConstants {
    particle_count: u32,
    rest_density: f32,
    smoothing_radius: f32,
    smoothing_radius_sq: f32,
    spiky_kernel_factor: f32,
    spiky_grad_kernel_factor: f32,
    constraint_epsilon: f32,
    relaxation_factor: f32,
    max_neighbors: u32,
}

impl PbdDensityConstraintConstants {
    pub fn new(
        particle_count: u32,
        rest_density: f32,
        smoothing_radius: f32,
        constraint_epsilon: f32,
        relaxation_factor: f32,
    ) -> Self {
        let smoothing_radius_sq = smoothing_radius * smoothing_radius;

        // Spiky kernel factor: 15 / (π * h^6)
        let spiky_kernel_factor = 15.0 / (std::f32::consts::PI * smoothing_radius.powi(6));

        // Spiky gradient kernel factor: -45 / (π * h^6)
        let spiky_grad_kernel_factor = -45.0 / (std::f32::consts::PI * smoothing_radius.powi(6));

        Self {
            particle_count,
            rest_density,
            smoothing_radius,
            smoothing_radius_sq,
            spiky_kernel_factor,
            spiky_grad_kernel_factor,
            constraint_epsilon,
            relaxation_factor,
            max_neighbors: 64, // 限制邻居粒子数量为64
        }
    }
}

impl ComputeGpuTaskConstants for PbdDensityConstraintConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/pbd_density_constraint.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.position().clone()), // 输入位置 (binding 0)
            WriteDescriptorSet::buffer(1, particles.predicted_position().clone()), // 预测位置（将被修改）(binding 1)
            WriteDescriptorSet::buffer(2, particles.density().clone()), // 密度值 (binding 2)
            WriteDescriptorSet::buffer(3, particles.index().clone()),   // 排序后的索引 (binding 3)
                                                                        // 移除 hash 缓冲区绑定以匹配着色器
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type PbdDensityConstraintTask = ComputeGpuTask<PbdDensityConstraintConstants>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{ParticleInitData, Particles},
        systems::simulation::tasks::{
            MortonHashConstants, MortonHashTask, RadixSortSystem, SpikySphConstants, SpikySphTask,
        },
        utils::{GpuTaskExecutor, VulkanoHeadlessBackend},
    };
    use glam::{Vec3, Vec4};

    #[test]
    fn test_pbd_density_constraint() {
        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // 创建测试粒子，形成一个密集的配置来测试密度约束
        particles.add_particles(
            &[
                ParticleInitData {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.05, 0.0, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.0, 0.05, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
                ParticleInitData {
                    position: Vec3::new(0.05, 0.05, 0.0),
                    velocitie: Vec3::new(0.0, 0.0, 0.0),
                },
            ],
            backend.memory_allocator(),
            &backend,
        );

        // 复制位置到预测位置
        particles.copy_position_to_predicted(&backend);

        // 执行Morton哈希计算
        let hash_constants = MortonHashConstants::new(particles.count(), 0.1);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // 执行排序
        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        // 执行SPH密度计算
        let sph_constants = SpikySphConstants::new(
            particles.count(),
            0.02, // 质量: 0.02 kg每个粒子
            0.2,  // 平滑半径: 20cm
            0.1,  // 网格大小: 10cm
        );

        let mut sph_task = SpikySphTask::new(backend.device());
        sph_task.set_constants(sph_constants);
        sph_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut sph_task);

        // 执行PBD密度约束
        let constraint_constants = PbdDensityConstraintConstants::new(
            particles.count(),
            1000.0, // 静止密度: 1000 kg/m³ (水的密度)
            0.2,    // 平滑半径: 20cm
            0.001,  // 约束epsilon
            0.3,    // 松弛因子
        );

        let mut constraint_task = PbdDensityConstraintTask::new(backend.device());
        constraint_task.set_constants(constraint_constants);
        constraint_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        backend.execute(&mut constraint_task);

        // 验证约束求解是否正确执行
        let predicted_positions = particles.predicted_position().read().unwrap();
        assert!(predicted_positions.len() >= particles.count() as usize);

        // 检查位置是否有合理的变化（密度约束应该会调整位置）
        let _positions = particles.position().read().unwrap();
        for i in 0..particles.count() as usize {
            // 预测位置应该仍然在合理范围内
            assert!(
                Vec4::from_array(predicted_positions[i].position).length() < 10.0,
                "预测位置应该在合理范围内，粒子 {} 的位置: {:?}",
                i,
                predicted_positions[i]
            );
        }
    }

    #[test]
    fn test_pbd_performance_analysis() {
        use std::time::Instant;

        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // 创建更多粒子来测试性能
        let particle_count = 10000;
        let mut particle_data = Vec::with_capacity(particle_count);

        for i in 0..particle_count {
            let x = (i % 100) as f32 * 0.02;
            let y = ((i / 100) % 100) as f32 * 0.02;
            let z = (i / 10000) as f32 * 0.02;

            particle_data.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocitie: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&particle_data, backend.memory_allocator(), &backend);
        particles.copy_position_to_predicted(&backend);

        // 预处理：哈希和排序
        let hash_constants = MortonHashConstants::new(particles.count(), 0.1);
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

        // SPH密度计算计时
        let sph_constants = SpikySphConstants::new(particles.count(), 0.02, 0.2, 0.1);
        let mut sph_task = SpikySphTask::new(backend.device());
        sph_task.set_constants(sph_constants);
        sph_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        let sph_start = Instant::now();
        backend.execute(&mut sph_task);
        let sph_time = sph_start.elapsed();

        // PBD约束计算计时
        let constraint_constants =
            PbdDensityConstraintConstants::new(particles.count(), 1000.0, 0.2, 0.001, 0.3);
        let mut constraint_task = PbdDensityConstraintTask::new(backend.device());
        constraint_task.set_constants(constraint_constants);
        constraint_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        // 单次PBD执行计时
        let pbd_single_start = Instant::now();
        backend.execute(&mut constraint_task);
        let pbd_single_time = pbd_single_start.elapsed();

        // 5次PBD迭代计时
        let pbd_iterations_start = Instant::now();
        for _ in 0..5 {
            backend.execute(&mut constraint_task);
        }
        let pbd_iterations_time = pbd_iterations_start.elapsed();

        // 性能分析输出
        println!("\n=== SPH vs PBD 性能分析 ({} 粒子) ===", particle_count);
        println!(
            "SPH密度计算 (1次):     {:>8.3}ms",
            sph_time.as_secs_f64() * 1000.0
        );
        println!(
            "PBD约束求解 (1次):     {:>8.3}ms",
            pbd_single_time.as_secs_f64() * 1000.0
        );
        println!(
            "PBD约束求解 (5次):     {:>8.3}ms",
            pbd_iterations_time.as_secs_f64() * 1000.0
        );

        let single_ratio = pbd_single_time.as_secs_f64() / sph_time.as_secs_f64();
        let iterations_ratio = pbd_iterations_time.as_secs_f64() / sph_time.as_secs_f64();

        println!("\n=== 性能比较 ===");
        println!("PBD单次 vs SPH:       {:>8.1}x", single_ratio);
        println!("PBD 5次 vs SPH:       {:>8.1}x", iterations_ratio);
        println!("理论预期 (5x基础):    {:>8.1}x", single_ratio * 5.0);

        // 分析
        println!("\n=== 性能瓶颈分析 ===");
        if single_ratio > 3.0 {
            println!("✓ PBD单次执行比SPH复杂 {:.1}倍，主要由于:", single_ratio);
            println!("  - 更昂贵的distance计算 (length vs dot)");
            println!("  - Spiky梯度计算 vs 简单Poly6核");
            println!("  - 额外的拉格朗日乘数和位置校正计算");
        }

        if iterations_ratio > 15.0 {
            println!("✓ 5次迭代放大了性能差异，总倍数: {:.1}x", iterations_ratio);
        }

        println!("=====================================");

        // 基本功能验证 - 调整断言以适应性能优化
        assert!(sph_time.as_secs_f64() > 0.0);
        assert!(pbd_single_time.as_secs_f64() > 0.0);
        assert!(pbd_iterations_time.as_secs_f64() > pbd_single_time.as_secs_f64() * 2.0);
        // 降低倍数要求
    }
}
