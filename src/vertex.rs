use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex, Debug, Clone)]
#[repr(C)]

pub struct EngineVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    // #[format(R32G32B32_SFLOAT)]
    // pub normal: [f32; 3],

    // #[format(R32G32_SFLOAT)]
    // pub uv: [f32; 2],
}
