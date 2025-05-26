use glam::Vec3;

pub(crate) struct SimulationConfig {
    pub gravity: Vec3,
    pub grid_size: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            grid_size: 0.1f32,
        }
    }
}
