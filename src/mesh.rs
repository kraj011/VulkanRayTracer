use vulkano::{
    acceleration_structure::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureBuildRangeInfo,
        AccelerationStructureGeometries, AccelerationStructureGeometryTrianglesData, GeometryFlags,
    },
    buffer::{IndexBuffer, Subbuffer},
    format::Format,
};

use crate::{material::Material, vertex::EngineVertex};

#[derive(Debug)]
pub struct Mesh {
    // raw points (not transformed)
    pub vertices: Vec<EngineVertex>,
    // array of indices
    pub indices: Vec<u32>,
    pub xform: glam::Mat4,
    pub name: String,

    pub material: Option<Material>,
    pub material_path: Option<String>,

    // vulkan
    pub vertex_buffer: Option<Subbuffer<[EngineVertex]>>,
    pub index_buffer: Option<IndexBuffer>,
    pub mvp: glam::Mat4,
}

impl Mesh {
    pub fn get_acceleration_structure(
        &self,
    ) -> (
        AccelerationStructureBuildRangeInfo,
        AccelerationStructureGeometries,
    ) {
        let vertex_max = self.indices.iter().max();
        let mut triangles =
            AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT);
        triangles.vertex_data = self.vertex_buffer.clone().map(|b| b.into_bytes());
        triangles.vertex_stride = std::mem::size_of::<EngineVertex>() as u32;
        triangles.max_vertex = *vertex_max.unwrap();
        triangles.index_data = self.index_buffer.clone();

        triangles.flags = GeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION | GeometryFlags::OPAQUE;

        let range = AccelerationStructureBuildRangeInfo {
            primitive_count: (self.indices.len() / 3) as u32,
            ..Default::default()
        };

        let mut triangles_vec = Vec::new();
        triangles_vec.push(triangles);

        (
            range,
            AccelerationStructureGeometries::Triangles(triangles_vec),
        )
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
            xform: glam::Mat4::IDENTITY,
            name: String::new(),

            material: None,
            material_path: None,

            vertex_buffer: None,
            index_buffer: None,
            mvp: glam::Mat4::IDENTITY,
        }
    }
}
