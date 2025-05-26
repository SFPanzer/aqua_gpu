use std::sync::Arc;

use vulkano::memory::allocator::StandardMemoryAllocator;

use super::particles::Particles;

pub(crate) struct ParticlePingPongBuffer {
    pub particles: [Box<Particles>; 2],
    pub current: usize,
}

impl ParticlePingPongBuffer {
    pub fn new(memory_allocator: &Arc<StandardMemoryAllocator>) -> Self {
        let particles = [
            Box::new(Particles::new(memory_allocator)),
            Box::new(Particles::new(memory_allocator)),
        ];
        Self {
            particles,
            current: 0,
        }
    }

    pub fn swap(&mut self) {
        self.current = (self.current + 1) % 2;
    }

    pub fn src(&self) -> &Particles {
        &self.particles[self.current]
    }

    pub fn dst(&mut self) -> &mut Particles {
        &mut self.particles[(self.current + 1) % 2]
    }
}
