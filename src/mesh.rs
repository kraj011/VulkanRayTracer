#[derive(Debug)]
pub struct Mesh {
    // raw points (not transformed)
    pub vertices: Vec<glam::Vec3>,
    // array of indices
    pub indices: Vec<i32>,
    pub xform: glam::Mat4,
    pub name: String,
}
