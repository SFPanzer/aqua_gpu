use std::sync::Arc;

use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::utils::GpuTaskExecutor;

use super::particles::Particles;

pub(crate) struct ParticlePingPongBuffer {
    src: Particles,
    dst: Particles,
}

impl ParticlePingPongBuffer {
    pub fn new(memory_allocator: &Arc<StandardMemoryAllocator>) -> Self {
        let src = Particles::new(memory_allocator);
        let dst = Particles::new(memory_allocator);
        Self {
            src,
            dst,
        }
    }

    pub fn swap(&mut self, task_executor: &impl GpuTaskExecutor) {
        self.src.replace_particles_from_particles(&self.dst, task_executor);
    }

    pub fn src(&self) -> &Particles {
        &self.src
    }

    pub fn dst(&mut self) -> &mut Particles {
        &mut self.dst
    }
}
