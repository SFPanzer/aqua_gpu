use std::{rc::Rc, sync::Arc, time::Instant};

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use crate::{core::Particles, utils::VulkanoBackend};

use super::{simulation_config::SimulationConfig, simulation_tasks::SimulationTasks};

pub(crate) struct SimulationSystem {
    vulkano_backend: Option<Rc<VulkanoBackend>>,
    tasks: Option<SimulationTasks>,
    config: SimulationConfig,
    last_update: Option<Instant>,
}

impl SimulationSystem {
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            vulkano_backend: None,
            tasks: None,
            config,
            last_update: None,
        }
    }

    pub fn init(&mut self, vulkano_backend: &Rc<VulkanoBackend>) {
        self.vulkano_backend = Some(vulkano_backend.clone());
        self.tasks = Some(SimulationTasks::new(vulkano_backend.device()));
    }

    pub fn update(
        &mut self,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        particles: &mut Particles,
    ) {
        let now = Instant::now();
        let dt = self
            .last_update
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32());
        self.last_update = Some(now);

        let tasks = self.tasks.as_mut().unwrap();
        tasks.set_constants(
            self.config.simulation_aabb,
            particles.count(),
            dt,
            self.config.gravity,
            self.config.grid_size,
        );
        tasks.update_descriptor_sets(descriptor_set_allocator, particles);
        tasks.execute(self.vulkano_backend.as_ref().unwrap().as_ref());
    }
}
