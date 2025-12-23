use glam::{Mat4, Vec3, Vec4};

use crate::{ray::Ray, sampling::random_in_unit_disk};

#[derive(Debug)]
pub struct Camera {
    pub focal_len: f32,
    pub sensor_x: f32,
    pub sensor_y: f32,

    pub width: u32,
    pub height: u32,
    pub sample_count: u32,

    pub xform: Mat4,
}

impl Camera {
    pub fn get_projection(&self) -> Mat4 {
        let aspect = self.width as f32 / self.height as f32;
        let fov = 2.0 * (self.sensor_y / (2.0 * self.focal_len)).atan();

        let mut proj = glam::Mat4::perspective_rh(fov, aspect, 0.1, 5000.0);
        proj.y_axis.y *= -1.0;
        proj
    }

    pub fn generate_ray(&self, u: f32, v: f32) -> Ray {
        let disk = random_in_unit_disk();
        let origin = (self.xform * Vec4::new(disk[0], disk[1], 0.0, 1.0)).truncate();
        let direction_local = Vec3::new(
            self.sensor_x * (u / (self.width as f32) - 0.5),
            self.sensor_y * (0.5 - (v / (self.height as f32))),
            -self.focal_len,
        );
        let direction_world = (self.xform * (direction_local - origin).extend(0.0)).truncate();

        Ray::new_default(origin, direction_world)
    }
}
