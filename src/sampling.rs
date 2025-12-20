use std::f32::consts::PI;

use rand::Rng;

pub fn random_in_unit_disk() -> [f32; 2] {
    let mut rng = rand::rng();
    let r = f32::sqrt(rng.random::<f32>());
    let phi = 2.0 * PI * rng.random::<f32>();
    let x = r * phi.cos();
    let y = r * phi.sin();
    return [x, y];
}
