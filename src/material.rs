use glam::Vec3;

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Vec3,
    pub emission: Option<Vec3>,
    pub name: String,
}