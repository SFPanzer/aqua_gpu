use std::{any::TypeId, collections::HashMap, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BufferCopy, CopyBufferInfoTyped, PrimaryAutoCommandBuffer,
    },
    descriptor_set::DescriptorSet,
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::{self, GpuFuture},
};

use crate::utils::{GpuTask, VulkanoBackend};

use super::particle_data::{ParticleHashEntry, ParticlePosition, ParticleVelocity};

pub(crate) type TaskId = TypeId;

const PARTICLE_MAX_COUNT: u32 = 0x100000; // 1 million particles

pub(crate) struct Particles {
    count: u32,
    cursor: u32,
    position: Subbuffer<[ParticlePosition]>,
    velocity: Subbuffer<[ParticleVelocity]>,
    hash: Subbuffer<[ParticleHashEntry]>,
    descriptor_sets: HashMap<TaskId, Arc<DescriptorSet>>,
}

impl Particles {
    pub fn new(vulkano_backend: &VulkanoBackend) -> Self {
        let position = Buffer::new_slice(
            vulkano_backend.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();
        let velocity = Buffer::new_slice(
            vulkano_backend.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();
        let hash = Buffer::new_slice(
            vulkano_backend.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            PARTICLE_MAX_COUNT as u64,
        )
        .unwrap();

        Self {
            position,
            velocity,
            hash,
            count: 0,
            cursor: 0,
            descriptor_sets: HashMap::new(),
        }
    }

    #[allow(unused)]
    pub fn particle_position_buffer(&self) -> &Subbuffer<[ParticlePosition]> {
        &self.position
    }

    #[allow(unused)]
    pub fn particle_velocity_buffer(&self) -> &Subbuffer<[ParticleVelocity]> {
        &self.velocity
    }

    #[allow(unused)]
    pub fn particle_hash_buffer(&self) -> &Subbuffer<[ParticleHashEntry]> {
        &self.hash
    }

    pub fn particle_count(&self) -> u32 {
        self.count
    }

    pub fn descriptor_sets(&mut self) -> &mut HashMap<TaskId, Arc<DescriptorSet>> {
        &mut self.descriptor_sets
    }

    pub fn add_particles(
        &mut self,
        particles: &[(ParticlePosition, ParticleVelocity)],
        vulkano_backend: &VulkanoBackend,
    ) {
        let regions = if (self.cursor + particles.len() as u32) < self.position.len() as u32 {
            vec![BufferCopy {
                src_offset: 0,
                dst_offset: self.cursor as u64,
                size: particles.len() as u64,
                ..Default::default()
            }]
        } else {
            let head_size = self.position.len() as u32 - self.cursor;
            let tail_size = particles.len() as u32 - head_size;
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
        self.replace_particles(particles, &regions, vulkano_backend);
        self.count = (self.count + particles.len() as u32).min(PARTICLE_MAX_COUNT);
    }

    pub fn replace_particles(
        &mut self,
        particles: &[(ParticlePosition, ParticleVelocity)],
        regions: &[BufferCopy],
        vulkano_backend: &VulkanoBackend,
    ) {
        let positions = particles.iter().map(|(p, _)| *p).collect::<Vec<_>>();
        let velocities = particles.iter().map(|(_, v)| *v).collect::<Vec<_>>();

        let stage_position_buffer = Buffer::from_iter(
            vulkano_backend.memory_allocator().clone(),
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
            vulkano_backend.memory_allocator().clone(),
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
        vulkano_backend.execute_gpu_task(&mut stage_task);
    }
}

struct ParticleStageTask {
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
