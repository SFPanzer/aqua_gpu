use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents, Vertex)]
pub(crate) struct ParticlePosition {
    #[format(R32G32B32A32_SFLOAT)]
    pub position: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, BufferContents, Vertex)]
pub(crate) struct ParticleVelocity {
    #[format(R32G32B32A32_SFLOAT)]
    pub velocity: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, BufferContents)]
pub(crate) struct ParticleHashEntry {
    pub hash: u32,
    pub index: u32,
}
#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{ParticlePosition, ParticleVelocity};

    impl ParticlePosition {
        pub fn new(position: Vec3) -> Self {
            Self {
                position: position.extend(0.0).into(),
            }
        }
    }

    impl ParticleVelocity {
        pub fn new(velocity: Vec3) -> Self {
            Self {
                velocity: velocity.extend(0.0).into(),
            }
        }
    }
}
