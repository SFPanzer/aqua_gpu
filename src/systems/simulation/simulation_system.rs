use std::{rc::Rc, sync::Arc};

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use crate::{core::Particles, utils::VulkanoBackend};

use super::simulation_tasks::MortonHashTask;

pub(crate) struct SimulationSystem {
    vulkano_backend: Option<Rc<VulkanoBackend>>,
    grid_size: f32,
    morton_task: Option<MortonHashTask>,
}

impl SimulationSystem {
    pub fn new() -> Self {
        Self {
            vulkano_backend: None,
            morton_task: None,
            grid_size: 1.0,
        }
    }

    pub fn init(&mut self, vulkano_backend: &Rc<VulkanoBackend>) {
        let morton_task = MortonHashTask::new(vulkano_backend.device());

        self.morton_task = Some(morton_task);
        self.vulkano_backend = Some(vulkano_backend.clone());
    }

    pub fn update(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        self.setup(descriptor_set_allocator, particles);

        let backend = self.vulkano_backend.as_mut().unwrap();
        backend.execute_gpu_task(self.morton_task.as_mut().unwrap());
    }

    fn setup(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
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
    }
}
