use glam::Vec3;
use vulkano::buffer::BufferContents;

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Vec3,
    pub emission: Option<Vec3>,
    pub name: String,

    pub engine_material: EngineMaterial,
}

#[repr(C)]
#[derive(Clone, Copy, BufferContents, Debug)]
pub struct EngineMaterial {
    pub albedo: [f32; 4],
    pub emission: [f32; 4],
}
