use std::{any::TypeId, collections::HashMap, sync::Arc};

use vulkano::{
    buffer::BufferContents,
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
    shader::EntryPoint,
    sync::{self, GpuFuture},
};

use crate::{
    core::{Particles, TaskId},
    utils::GpuTask,
};

pub(crate) trait ComputeGpuTaskConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint;
    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet>;
    fn particle_count(&self) -> u32;
}

pub(crate) struct ComputeGpuTask<C>
where
    C: BufferContents + ComputeGpuTaskConstants,
{
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    constants: Option<C>,
}

impl<C> ComputeGpuTask<C>
where
    C: BufferContents + ComputeGpuTaskConstants,
{
    pub fn new(device: &Arc<Device>) -> Self {
        let entry_point = C::entry_point(device);
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

    pub fn set_constants(&mut self, constants: C) {
        self.constants = Some(constants);
    }

    pub fn update_descriptor_set(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        if let Err(_) = self.try_bind_descriptor_set_from_cache(particles.descriptor_sets()) {
            self.create_and_bind_descriptor_set(descriptor_set_allocator, particles)
        }
    }

    fn try_bind_descriptor_set_from_cache(
        &mut self,
        descriptor_sets: &mut HashMap<TaskId, Arc<DescriptorSet>>,
    ) -> Result<(), ()> {
        let task_id = TypeId::of::<Self>();
        if let Some(descriptor_set) = descriptor_sets.get(&task_id) {
            self.descriptor_set = Some(descriptor_set.clone());
            Ok(())
        } else {
            Err(())
        }
    }

    fn create_and_bind_descriptor_set(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        let task_id = TypeId::of::<Self>();

        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            layout.clone(),
            C::descriptor_writes(particles),
            [],
        )
        .unwrap();
        particles
            .descriptor_sets()
            .insert(task_id, descriptor_set.clone());

        self.descriptor_set = Some(descriptor_set)
    }
}

impl<C> GpuTask for ComputeGpuTask<C>
where
    C: BufferContents + Copy + ComputeGpuTaskConstants,
{
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
        let work_group_num = self.constants.as_ref().unwrap().particle_count() / 256 + 1;
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
