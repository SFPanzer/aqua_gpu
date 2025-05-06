use std::{cmp::min, sync::Arc};

use glam::Vec3;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BufferCopy, CopyBufferInfoTyped, PrimaryAutoCommandBuffer,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::graphics::vertex_input::Vertex,
    sync::{self, GpuFuture},
};

use crate::utils::{GpuTask, VulkanoBackend};

const PARTICLE_MAX_COUNT: u32 = 0x1000;

pub(crate) struct Particles {
    buffer: Subbuffer<[Particle]>,
    count: u32,
    cursor: u32,
}

impl Particles {
    pub fn new(vulkano_backend: &VulkanoBackend) -> Self {
        let particles = Buffer::new_slice(
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

        Self {
            buffer: particles,
            count: 0,
            cursor: 0,
        }
    }

    pub fn particle_buffer(&self) -> &Subbuffer<[Particle]> {
        &self.buffer
    }

    pub fn particle_count(&self) -> u32 {
        self.count
    }

    pub fn add_particles(&mut self, particles: &[Particle], vulkano_backend: &VulkanoBackend) {
        self.count = min(
            self.count + particles.len() as u32,
            self.buffer.len() as u32,
        );

        let regions = if (self.cursor + particles.len() as u32) < self.buffer.len() as u32 {
            vec![BufferCopy {
                src_offset: 0,
                dst_offset: self.cursor as u64,
                size: particles.len() as u64,
                ..Default::default()
            }]
        } else {
            let head_size = self.buffer.len() as u32 - self.cursor;
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
    }

    fn replace_particles(
        &mut self,
        particles: &[Particle],
        regions: &[BufferCopy],
        vulkano_backend: &VulkanoBackend,
    ) {
        let mut task = ParticleStageTask::new(particles, regions.to_vec(), vulkano_backend);
        vulkano_backend.execute_gpu_task(&mut task, self);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents, Vertex)]
pub(crate) struct Particle {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

impl Particle {
    pub fn new(position: Vec3) -> Self {
        Self {
            position: position.into(),
        }
    }
}
struct ParticleStageTask {
    staging: Subbuffer<[Particle]>,
    regions: Vec<BufferCopy>,
}

impl ParticleStageTask {
    pub fn new(
        particles: &[Particle],
        regions: Vec<BufferCopy>,
        vulkano_backend: &VulkanoBackend,
    ) -> Self {
        let staging = Buffer::from_iter(
            vulkano_backend.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            particles.iter().cloned(),
        )
        .unwrap();

        Self { staging, regions }
    }
}

impl GpuTask for ParticleStageTask {
    fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        particles: &Particles,
    ) {
        let mut copy_buffer_info =
            CopyBufferInfoTyped::buffers(self.staging.clone(), particles.buffer.clone());
        copy_buffer_info.regions = self.regions.clone().into();
        builder.copy_buffer(copy_buffer_info).unwrap();
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
