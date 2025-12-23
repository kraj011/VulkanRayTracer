use vulkano::buffer::Subbuffer;

use crate::vertex::EngineVertex;

#[derive(Debug)]
pub struct Mesh {
    // raw points (not transformed)
    pub vertices: Vec<EngineVertex>,
    // array of indices
    pub indices: Vec<u32>,
    pub xform: glam::Mat4,
    pub name: String,

    // vulkan
    pub vertex_buffer: Option<Subbuffer<[EngineVertex]>>,
    pub index_buffer: Option<Subbuffer<[u32]>>,
    pub mvp: glam::Mat4,
}

impl Default for Mesh {
    fn default() -> Self {
        Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
            xform: glam::Mat4::IDENTITY,
            name: String::new(),

            vertex_buffer: None,
            index_buffer: None,
            mvp: glam::Mat4::IDENTITY,
        }
    }
}
