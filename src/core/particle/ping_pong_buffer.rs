use crate::utils::VulkanoBackend;

use super::particles::Particles;

pub(crate) struct ParticlePingPongBuffer {
    pub particles: [Box<Particles>; 2],
    pub current: usize,
}

impl ParticlePingPongBuffer {
    pub fn new(vulkano_backend: &VulkanoBackend) -> Self {
        let particles = [
            Box::new(Particles::new(vulkano_backend)),
            Box::new(Particles::new(vulkano_backend)),
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
