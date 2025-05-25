use std::{any::TypeId, collections::HashMap, sync::Arc};

use glam::Vec3;
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
    core::{ParticleVelocity, TaskId},
    utils::GpuTask,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents)]
struct ApplyGravityConstants {
    particle_count: u32,
    dt: f32,
    _padding: [f32; 2],
    gravity: [f32; 3],
}

pub(crate) struct ApplyGravityTask {
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    constants: Option<ApplyGravityConstants>,
}

impl ApplyGravityTask {
    pub fn new(device: &Arc<Device>) -> Self {
        mod apply_gravity {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/apply_gravity.comp",
            }
        }
        let entry_point = apply_gravity::load(device.clone())
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

    pub fn set_constants(&mut self, particle_count: u32, dt: f32, gravity: Vec3) {
        self.constants = Some(ApplyGravityConstants {
            particle_count,
            dt,
            gravity: gravity.to_array(),
            _padding: [0., 0.],
        });
    }

    pub fn set_descriptor_set(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        descriptor_sets: &mut HashMap<TaskId, Arc<DescriptorSet>>,
        velocity: &Subbuffer<[ParticleVelocity]>,
    ) {
        let task_id = TypeId::of::<Self>();

        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::buffer(0, velocity.clone())],
            [],
        )
        .unwrap();
        descriptor_sets.insert(task_id, descriptor_set.clone());

        self.descriptor_set = Some(descriptor_set)
    }
}

impl GpuTask for ApplyGravityTask {
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
fn test_apply_gravity() {
    use crate::utils::approx_eq;
    use crate::utils::VulkanoHeadlessBackend;
    use glam::Vec3;
    use vulkano::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    };

    let backend = VulkanoHeadlessBackend::new();

    let mut task = ApplyGravityTask::new(backend.device());

    let velocities = vec![
        ParticleVelocity::new(Vec3::new(-1., 0., 0.)),
        ParticleVelocity::new(Vec3::new(0., -1., 0.)),
        ParticleVelocity::new(Vec3::new(0., 0., -1.)),
    ];
    let velocities_buffer: Subbuffer<[_]> = Buffer::from_iter(
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

    task.set_constants(velocities.len() as u32, 0.02, Vec3::new(10., 20., 30.));
    task.set_descriptor_set(
        &backend.descriptor_set_allocator(),
        &mut descriptor_sets,
        &velocities_buffer,
    );

    backend.execute_gpu_task(&mut task);

    let result_entries = velocities_buffer.read().unwrap();
    let expected_entries = vec![
        ParticleVelocity::new(Vec3::new(-0.8, 0.4, 0.6)),
        ParticleVelocity::new(Vec3::new(0.2, -0.6, 0.6)),
        ParticleVelocity::new(Vec3::new(0.2, 0.4, -0.4)),
    ];
    assert_eq!(result_entries.len(), expected_entries.len());
    for (r, e) in result_entries.iter().zip(expected_entries.iter()) {
        for (i, (a, b)) in r.velocity.iter().zip(e.velocity.iter()).enumerate() {
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
