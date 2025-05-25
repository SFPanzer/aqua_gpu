use std::{any::TypeId, collections::HashMap, sync::Arc};

use vulkano::{
    buffer::{BufferContents, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
};

use crate::{
    core::{ParticlePosition, ParticleVelocity, TaskId},
    utils::GpuTask,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents)]
struct UpdatePositionConstants {
    particle_count: u32,
    dt: f32,
}

pub(crate) struct UpdatePositionTask {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    constants: Option<UpdatePositionConstants>,
}

impl UpdatePositionTask {
    pub fn new(device: &Arc<Device>) -> Self {
        mod update_position {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/update_position.comp",
            }
        }
        let entry_point = update_position::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        Self {
            pipeline,
            descriptor_set: None,
            constants: None,
        }
    }

    pub fn update_descriptor_set(
        &mut self,
        descriptor_sets: &HashMap<TaskId, Arc<DescriptorSet>>,
    ) -> Result<(), ()> {
        let task_id = TypeId::of::<Self>();
        if let Some(descriptor_set) = descriptor_sets.get(&task_id) {
            self.descriptor_set = Some(descriptor_set.clone());
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn set_constants(&mut self, particle_count: u32, dt: f32) {
        self.constants = Some(UpdatePositionConstants { particle_count, dt });
    }

    pub fn set_descriptor_set(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        descriptor_sets: &mut HashMap<TaskId, Arc<DescriptorSet>>,
        positions: &Subbuffer<[ParticlePosition]>,
        velocity: &Subbuffer<[ParticleVelocity]>,
    ) {
        let task_id = TypeId::of::<Self>();

        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, velocity.clone()),
                WriteDescriptorSet::buffer(1, positions.clone()),
            ],
            [],
        )
        .unwrap();
        descriptor_sets.insert(task_id, descriptor_set.clone());

        self.descriptor_set = Some(descriptor_set)
    }
}

impl GpuTask for UpdatePositionTask {
    fn record(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap();
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                self.descriptor_set.as_ref().unwrap().clone(),
            )
            .unwrap();
        builder
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                *self.constants.as_ref().unwrap(),
            )
            .unwrap();
        let work_group_num = self.constants.as_ref().unwrap().particle_count / 256 + 1;
        unsafe {
            builder.dispatch([work_group_num, 1, 1]).unwrap();
        }
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

#[test]
fn test_update_position() {
    use crate::utils::approx_eq;
    use crate::utils::VulkanoHeadlessBackend;
    use glam::Vec3;
    use vulkano::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    };

    let backend = VulkanoHeadlessBackend::new();

    let mut task = UpdatePositionTask::new(backend.device());

    let positions = vec![
        ParticlePosition::new(Vec3::new(-1., 0., 0.)),
        ParticlePosition::new(Vec3::new(0., -1., 0.)),
        ParticlePosition::new(Vec3::new(0., 0., -1.)),
    ];
    let positions_buffer = Buffer::from_iter(
        backend.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        positions.iter().cloned(),
    )
    .unwrap();

    let velocities = vec![
        ParticleVelocity::new(Vec3::new(-1., 0., 0.)),
        ParticleVelocity::new(Vec3::new(0., -1., 0.)),
        ParticleVelocity::new(Vec3::new(0., 0., -1.)),
    ];
    let velocities_buffer = Buffer::from_iter(
        backend.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        velocities.iter().cloned(),
    )
    .unwrap();

    let mut descriptor_sets = HashMap::new();

    task.set_constants(positions.len() as u32, 1.0);
    task.set_descriptor_set(
        &backend.descriptor_set_allocator(),
        &mut descriptor_sets,
        &positions_buffer,
        &velocities_buffer,
    );

    backend.execute_gpu_task(&mut task);

    let result_entries = positions_buffer.read().unwrap();
    let expected_entries = vec![
        ParticlePosition::new(Vec3::new(-2., 0., 0.)),
        ParticlePosition::new(Vec3::new(0., -2., 0.)),
        ParticlePosition::new(Vec3::new(0., 0., -2.)),
    ];
    assert_eq!(result_entries.len(), expected_entries.len());
    for (r, e) in result_entries.iter().zip(expected_entries.iter()) {
        for (i, (a, b)) in r.position.iter().zip(e.position.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Mismatch at component {}: {} != {}",
                i,
                a,
                b
            );
        }
    }
}
