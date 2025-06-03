use glam::Vec3;

use crate::core::Aabb;

pub(crate) struct SimulationConfig {
    pub simulation_aabb: Aabb,
    pub gravity: Vec3,
    pub grid_size: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            simulation_aabb: Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            grid_size: 0.1f32,
        }
    }
}
