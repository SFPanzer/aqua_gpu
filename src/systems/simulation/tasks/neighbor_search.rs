use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::Particles;

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

/// 邻居搜索任务常量
/// 基于博客中的邻域搜索实现，用于填充contacts和contact_counts缓冲区
#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct NeighborSearchConstants {
    particle_count: u32,
    smoothing_radius: f32,
    smoothing_radius_sq: f32,
    grid_size: f32,
    max_neighbors: u32,
}

impl NeighborSearchConstants {
    pub fn new(
        particle_count: u32,
        smoothing_radius: f32,
        grid_size: f32,
        max_neighbors: u32,
    ) -> Self {
        let smoothing_radius_sq = smoothing_radius * smoothing_radius;

        Self {
            particle_count,
            smoothing_radius,
            smoothing_radius_sq,
            grid_size,
            max_neighbors,
        }
    }
}

impl ComputeGpuTaskConstants for NeighborSearchConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/neighbor_search.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.predicted_position().clone()), // predicted_positions
            WriteDescriptorSet::buffer(1, particles.index().clone()),              // sorted_indices
            WriteDescriptorSet::buffer(2, particles.cell_start().clone()),         // cell_start
            WriteDescriptorSet::buffer(3, particles.cell_end().clone()),           // cell_end
            WriteDescriptorSet::buffer(4, particles.contacts().clone()),           // contacts
            WriteDescriptorSet::buffer(5, particles.contact_counts().clone()),     // contact_counts
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type NeighborSearchTask = ComputeGpuTask<NeighborSearchConstants>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{ParticleInitData, Particles},
        systems::simulation::tasks::{
            BuildCellIndexConstants, BuildCellIndexTask, MortonHashConstants, MortonHashTask,
            PredictPositionConstants, PredictPositionTask, RadixSortSystem,
        },
        utils::{GpuTaskExecutor, VulkanoHeadlessBackend},
    };
    use glam::Vec3;

    #[test]
    fn test_neighbor_search() {
        let backend = VulkanoHeadlessBackend::new();
        let mut particles = Particles::new(backend.memory_allocator());

        // 创建测试粒子 - 密集排列以确保有邻居
        let mut test_particles = Vec::new();
        for i in 0..100 {
            let x = (i % 10) as f32 * 0.1;
            let y = (i / 10) as f32 * 0.1;
            let z = 0.0;
            test_particles.push(ParticleInitData {
                position: Vec3::new(x, y, z),
                velocity: Vec3::new(0.0, 0.0, 0.0),
            });
        }

        particles.add_particles(&test_particles, backend.memory_allocator(), &backend);

        // 1. 预测位置
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

        // 2. Morton哈希
        let hash_constants = MortonHashConstants::new(particles.count(), 0.2);
        let mut hash_task = MortonHashTask::new(backend.device());
        hash_task.set_constants(hash_constants);
        hash_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut hash_task);

        // 3. 排序
        let mut sort_system = RadixSortSystem::new(backend.device());
        sort_system.sort_morton_codes(
            &mut particles,
            &backend.descriptor_set_allocator(),
            &backend,
        );

        // 4. 构建cell索引
        let cell_index_constants = BuildCellIndexConstants::new(particles.count());
        let mut cell_index_task = BuildCellIndexTask::new(backend.device());
        cell_index_task.set_constants(cell_index_constants);
        cell_index_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);
        backend.execute(&mut cell_index_task);

        // 5. 执行邻居搜索
        let neighbor_constants = NeighborSearchConstants::new(
            particles.count(),
            0.2, // smoothing_radius
            0.2, // grid_size
            96,  // max_neighbors
        );

        let mut neighbor_task = NeighborSearchTask::new(backend.device());
        neighbor_task.set_constants(neighbor_constants);
        neighbor_task.update_descriptor_set(&backend.descriptor_set_allocator(), &mut particles);

        backend.execute(&mut neighbor_task);

        // 验证结果
        let contact_counts = particles.contact_counts().read().unwrap();
        let contacts = particles.contacts().read().unwrap();

        let mut total_neighbors = 0;
        let mut particles_with_neighbors = 0;

        for i in 0..particles.count() as usize {
            let neighbor_count = contact_counts[i];
            if neighbor_count > 0 {
                particles_with_neighbors += 1;
                total_neighbors += neighbor_count;

                // 验证邻居索引的有效性
                for n in 0..neighbor_count as usize {
                    let contact_index = i * 96 + n;
                    let neighbor_id = contacts[contact_index];
                    assert!(
                        neighbor_id < particles.count(),
                        "邻居索引超出范围: {} >= {}",
                        neighbor_id,
                        particles.count()
                    );
                    assert_ne!(neighbor_id, i as u32, "粒子不应该是自己的邻居");
                }
            }
        }

        println!("邻居搜索测试结果:");
        println!("总粒子数: {}", particles.count());
        println!("有邻居的粒子数: {}", particles_with_neighbors);
        println!("总邻居关系数: {}", total_neighbors);

        if particles_with_neighbors > 0 {
            println!(
                "平均邻居数: {:.2}",
                total_neighbors as f32 / particles_with_neighbors as f32
            );
        }

        assert!(particles_with_neighbors > 0, "应该有一些粒子找到邻居");
    }
}
