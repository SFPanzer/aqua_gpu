use std::sync::Arc;

use glam::Vec3;
use vulkano::{descriptor_set::allocator::StandardDescriptorSetAllocator, device::Device};

use crate::{core::Particles, utils::GpuTaskExecutor};

use super::tasks::{
    ApplyGravityConstants, ApplyGravityTask, MortonHashConstants, MortonHashTask,
    UpdatePositionConstants, UpdatePositionTask,
};

pub(crate) struct SimulationTasks {
    pub apply_gravity: ApplyGravityTask,
    pub morton_hash: MortonHashTask,
    pub update_position: UpdatePositionTask,
}

impl SimulationTasks {
    pub fn new(device: &Arc<Device>) -> Self {
        let apply_gravity = ApplyGravityTask::new(device);
        let morton_hash = MortonHashTask::new(device);
        let update_position = UpdatePositionTask::new(device);

        Self {
            apply_gravity,
            morton_hash,
            update_position,
        }
    }

    pub fn set_constants(&mut self, particle_count: u32, dt: f32, gravity: Vec3, grid_size: f32) {
        let apply_gravity_constants = ApplyGravityConstants::new(particle_count, dt, gravity);
        self.apply_gravity.set_constants(apply_gravity_constants);

        let morton_hash_constants = MortonHashConstants::new(particle_count, grid_size);
        self.morton_hash.set_constants(morton_hash_constants);

        let update_position_constants = UpdatePositionConstants::new(particle_count, dt);
        self.update_position
            .set_constants(update_position_constants);
    }

    pub fn update_descriptor_sets(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        self.apply_gravity
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.morton_hash
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.update_position
            .update_descriptor_set(descriptor_set_allocator, particles);
    }

    pub fn execute(&mut self, executor: &impl GpuTaskExecutor) {
        executor.execute(&mut self.apply_gravity);
        executor.execute(&mut self.morton_hash);
        executor.execute(&mut self.update_position);
    }
}
