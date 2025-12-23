use std::f32;

use glam::Vec3;

pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,

    pub min_t: f32,
    pub max_t: f32,
}

impl Ray {
    pub fn new(o: Vec3, d: Vec3, min_t: f32, max_t: f32) -> Self {
        Self { o, d, min_t, max_t }
    }

    pub fn new_default(o: Vec3, d: Vec3) -> Self {
        Self {
            o,
            d,
            min_t: f32::EPSILON,
            max_t: f32::INFINITY,
        }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        return self.o + self.d * t;
    }
}
