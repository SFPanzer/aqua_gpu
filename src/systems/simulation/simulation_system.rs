use std::{rc::Rc, sync::Arc};

use glam::Vec3;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use crate::{core::Particles, utils::VulkanoBackend};

use super::simulation_tasks::{ApplyGravityTask, MortonHashTask, UpdatePositionTask};

pub(crate) struct SimulationSystem {
    vulkano_backend: Option<Rc<VulkanoBackend>>,
    grid_size: f32,
    morton_task: Option<MortonHashTask>,
    apply_gravity: Option<ApplyGravityTask>,
    update_position: Option<UpdatePositionTask>,
}

impl SimulationSystem {
    pub fn new() -> Self {
        Self {
            vulkano_backend: None,
            morton_task: None,
            apply_gravity: None,
            update_position: None,
            grid_size: 1.0,
        }
    }

    pub fn init(&mut self, vulkano_backend: &Rc<VulkanoBackend>) {
        let morton_task = MortonHashTask::new(vulkano_backend.device());
        let apply_gravity = ApplyGravityTask::new(vulkano_backend.device());
        let update_position = UpdatePositionTask::new(vulkano_backend.device());

        self.morton_task = Some(morton_task);
        self.apply_gravity = Some(apply_gravity);
        self.update_position = Some(update_position);
        self.vulkano_backend = Some(vulkano_backend.clone());
    }

    pub fn update(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        self.setup(descriptor_set_allocator, particles);

        let backend = self.vulkano_backend.as_mut().unwrap();
        backend.execute_gpu_task(self.apply_gravity.as_mut().unwrap());
        backend.execute_gpu_task(self.morton_task.as_mut().unwrap());
        backend.execute_gpu_task(self.update_position.as_mut().unwrap());
    }

    fn setup(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        // Setup the `Apply gravity task`
        let apply_gravity = self.apply_gravity.as_mut().unwrap();
        apply_gravity.set_constants(particles.particle_count(), 0.02, Vec3::new(0., -9.8, 0.));
        if let Err(()) = apply_gravity.update_descriptor_set(particles.descriptor_sets()) {
            let velocity = particles.particle_velocity_buffer().clone();
            let descriptor_sets = particles.descriptor_sets();
            apply_gravity.set_descriptor_set(descriptor_set_allocator, descriptor_sets, &velocity);
        }
        // Setup the Morton hash task
        let morton_task = self.morton_task.as_mut().unwrap();
        morton_task.set_constants(particles.particle_count(), self.grid_size);
        if let Err(()) = morton_task.update_descriptor_set(particles.descriptor_sets()) {
            let positions = particles.particle_position_buffer().clone();
            let hash = particles.particle_hash_buffer().clone();
            let descriptor_sets = particles.descriptor_sets();
            morton_task.set_descriptor_set(
                descriptor_set_allocator,
                descriptor_sets,
                &positions,
                &hash,
            );
        }
        // Setup update position task
        let update_position = self.update_position.as_mut().unwrap();
        update_position.set_constants(particles.particle_count(), 0.02);
        if let Err(()) = update_position.update_descriptor_set(particles.descriptor_sets()) {
            let positions = particles.particle_position_buffer().clone();
            let velocity = particles.particle_velocity_buffer().clone();
            let descriptor_sets = particles.descriptor_sets();
            update_position.set_descriptor_set(
                descriptor_set_allocator,
                descriptor_sets,
                &positions,
                &velocity,
            );
        }
    }
}
