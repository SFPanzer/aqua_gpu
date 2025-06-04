use std::{any::TypeId, collections::HashMap, sync::Arc};

use glam::Vec3;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BufferCopy, CopyBufferInfoTyped, PrimaryAutoCommandBuffer,
    },
    descriptor_set::DescriptorSet,
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{self, GpuFuture},
};

use crate::utils::{GpuTask, GpuTaskExecutor};

use super::particle_data::{ParticlePosition, ParticleVelocity};

pub(crate) type TaskId = TypeId;

const PARTICLE_MAX_COUNT: u32 = 0x100000; // 1 million particles

pub struct ParticleInitData {
    pub position: Vec3,
    pub velocitie: Vec3,
}

pub(crate) struct Particles {
    count: u32,
    cursor: u32,
    position: Subbuffer<[ParticlePosition]>,
    velocity: Subbuffer<[ParticleVelocity]>,
    hash: Subbuffer<[u32]>,
    index: Subbuffer<[u32]>,
    hash_temp: Subbuffer<[u32]>,
    index_temp: Subbuffer<[u32]>,
    histograms: Subbuffer<[u32]>,
    prefix_sums: Subbuffer<[u32]>,
    density: Subbuffer<[f32]>,
    predicted_position: Subbuffer<[ParticlePosition]>,
    descriptor_sets: HashMap<TaskId, Arc<DescriptorSet>>,
}

impl Particles {
    pub fn new(memory_allocator: &Arc<StandardMemoryAllocator>) -> Self {
        let allocation_create_info = {
            let memory_type_filter = {
                #[cfg(test)]
                {
                    MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS
                }

                #[cfg(not(test))]
                {
                    MemoryTypeFilter::PREFER_DEVICE
                }
            };
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            }
        };

        let position = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        let velocity = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        let hash = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        let index = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        let hash_temp = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        let index_temp = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        let histograms = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        let prefix_sums = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        // SPH related buffers
        let density = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        // 新增: 初始化predicted_position缓冲区
        let predicted_position = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST, // 允许作为复制目标
                ..Default::default()
            },
            allocation_create_info.clone(),
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        Self {
            position,
            velocity,
            hash,
            index,
            hash_temp,
            index_temp,
            histograms,
            prefix_sums,
            density,
            predicted_position, // 新增
            count: 0,
            cursor: 0,
            descriptor_sets: HashMap::new(),
        }
    }

    #[allow(unused)]
    pub fn position(&self) -> &Subbuffer<[ParticlePosition]> {
        &self.position
    }

    #[allow(unused)]
    pub fn velocity(&self) -> &Subbuffer<[ParticleVelocity]> {
        &self.velocity
    }

    #[allow(unused)]
    pub fn hash(&self) -> &Subbuffer<[u32]> {
        &self.hash
    }

    pub fn index(&self) -> &Subbuffer<[u32]> {
        &self.index
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn histograms(&self) -> &Subbuffer<[u32]> {
        &self.histograms
    }

    pub fn hash_temp(&self) -> &Subbuffer<[u32]> {
        &self.hash_temp
    }

    pub fn index_temp(&self) -> &Subbuffer<[u32]> {
        &self.index_temp
    }

    pub fn prefix_sums(&self) -> &Subbuffer<[u32]> {
        &self.prefix_sums
    }

    // SPH related buffer accessors
    pub fn density(&self) -> &Subbuffer<[f32]> {
        &self.density
    }

    // 新增: predicted_position访问器
    #[allow(unused)]
    pub fn predicted_position(&self) -> &Subbuffer<[ParticlePosition]> {
        &self.predicted_position
    }

    pub fn descriptor_sets(&mut self) -> &mut HashMap<TaskId, Arc<DescriptorSet>> {
        &mut self.descriptor_sets
    }

    /// Swap main hash buffer and temporary hash buffer
    #[allow(unused)]
    pub fn swap_hash_buffers(&mut self) {
        std::mem::swap(&mut self.hash, &mut self.hash_temp);
    }

    /// Swap main index buffer and temporary index buffer
    #[allow(unused)]
    pub fn swap_index_buffers(&mut self) {
        std::mem::swap(&mut self.index, &mut self.index_temp);
    }

    pub fn add_particles(
        &mut self,
        particles_init_data: &[ParticleInitData],
        memory_allocator: &Arc<StandardMemoryAllocator>,
        task_executor: &dyn GpuTaskExecutor,
    ) {
        let regions =
            if (self.cursor + particles_init_data.len() as u32) < self.position.len() as u32 {
                vec![BufferCopy {
                    src_offset: 0,
                    dst_offset: self.cursor as u64,
                    size: particles_init_data.len() as u64,
                    ..Default::default()
                }]
            } else {
                let head_size = self.position.len() as u32 - self.cursor;
                let tail_size = particles_init_data.len() as u32 - head_size;
                vec![
                    BufferCopy {
                        src_offset: 0,
                        dst_offset: self.cursor as u64,
                        size: head_size as u64,
                        ..Default::default()
                    },
                    BufferCopy {
                        src_offset: head_size as u64,
                        dst_offset: 0,
                        size: tail_size as u64,
                        ..Default::default()
                    },
                ]
            };
        self.replace_particles_from_init_data(
            particles_init_data,
            &regions,
            memory_allocator,
            task_executor,
        );
        self.count = (self.count + particles_init_data.len() as u32).min(PARTICLE_MAX_COUNT);
    }

    pub fn replace_particles_from_init_data(
        &mut self,
        particles_init_data: &[ParticleInitData],
        regions: &[BufferCopy],
        memory_allocator: &Arc<StandardMemoryAllocator>,
        task_executor: &dyn GpuTaskExecutor,
    ) {
        let positions = particles_init_data
            .iter()
            .map(|p| ParticlePosition {
                position: p.position.extend(0.0).to_array(),
            })
            .collect::<Vec<_>>();
        let velocities = particles_init_data
            .iter()
            .map(|p| ParticleVelocity {
                velocity: p.velocitie.extend(0.0).to_array(),
            })
            .collect::<Vec<_>>();

        let stage_position_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            positions.iter().cloned(),
        )
        .unwrap();
        let stage_velocity_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            velocities.iter().cloned(),
        )
        .unwrap();

        let mut stage_task = ParticleStageTask::new(
            stage_position_buffer,
            stage_velocity_buffer,
            self.position.clone(),
            self.velocity.clone(),
            regions.to_vec(),
        );
        task_executor.execute(&mut stage_task);
    }

    pub fn replace_particles_from_particles(
        &mut self,
        src: &Self,
        task_executor: &impl GpuTaskExecutor,
    ) {
        self.count = src.count;
        self.cursor = src.cursor;

        if src.count() == 0 {
            return; // No particles to swap
        }

        let regions = [BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: src.count() as u64,
            ..Default::default()
        }];
        let mut swap_task = ParticleStageTask::new(
            src.position.clone(),
            src.velocity.clone(),
            self.position.clone(),
            self.velocity.clone(),
            regions.to_vec(),
        );
        task_executor.execute(&mut swap_task);
    }

    // 新增: 将position复制到predicted_position
    pub fn copy_position_to_predicted(&mut self, task_executor: &impl GpuTaskExecutor) {
        if self.count == 0 {
            return;
        }

        let regions = [BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: self.count() as u64,
            ..Default::default()
        }];

        let mut copy_task = PositionCopyTask::new(
            self.position.clone(),
            self.predicted_position.clone(),
            regions.to_vec(),
        );
        task_executor.execute(&mut copy_task);
    }
}

pub(super) struct ParticleStageTask {
    position_src: Subbuffer<[ParticlePosition]>,
    velocity_src: Subbuffer<[ParticleVelocity]>,
    position_dst: Subbuffer<[ParticlePosition]>,
    velocity_dst: Subbuffer<[ParticleVelocity]>,
    regions: Vec<BufferCopy>,
}

impl ParticleStageTask {
    pub fn new(
        position_src: Subbuffer<[ParticlePosition]>,
        velocity_src: Subbuffer<[ParticleVelocity]>,
        position_dst: Subbuffer<[ParticlePosition]>,
        velocity_dst: Subbuffer<[ParticleVelocity]>,
        regions: Vec<BufferCopy>,
    ) -> Self {
        Self {
            position_src,
            velocity_src,
            position_dst,
            velocity_dst,
            regions,
        }
    }
}

impl GpuTask for ParticleStageTask {
    fn record(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        let mut copy_positions_info =
            CopyBufferInfoTyped::buffers(self.position_src.clone(), self.position_dst.clone());
        let mut copy_velocities_info =
            CopyBufferInfoTyped::buffers(self.velocity_src.clone(), self.velocity_dst.clone());
        copy_positions_info.regions = self.regions.clone().into();
        copy_velocities_info.regions = self.regions.clone().into();

        builder.copy_buffer(copy_positions_info).unwrap();
        builder.copy_buffer(copy_velocities_info).unwrap();
    }

    fn submit(
        &mut self,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        queue: &Arc<Queue>,
        device: &Arc<Device>,
    ) {
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }
}

// 新增: PositionCopyTask，用于在GPU上复制位置数据
pub(super) struct PositionCopyTask {
    src: Subbuffer<[ParticlePosition]>,
    dst: Subbuffer<[ParticlePosition]>,
    regions: Vec<BufferCopy>,
}

impl PositionCopyTask {
    pub fn new(
        src: Subbuffer<[ParticlePosition]>,
        dst: Subbuffer<[ParticlePosition]>,
        regions: Vec<BufferCopy>,
    ) -> Self {
        Self { src, dst, regions }
    }
}

impl GpuTask for PositionCopyTask {
    fn record(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        let mut copy_info = CopyBufferInfoTyped::buffers(self.src.clone(), self.dst.clone());
        copy_info.regions = self.regions.clone().into();
        builder.copy_buffer(copy_info).unwrap();
    }

    fn submit(
        &mut self,
        command_buffer: Arc<PrimaryAutoCommandBuffer>,
        queue: &Arc<Queue>,
        device: &Arc<Device>,
    ) {
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }
}
