use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub(crate) struct RenderVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl RenderVertex {
    pub fn new(position: [f32; 2]) -> Self {
        Self { position }
    }
}
