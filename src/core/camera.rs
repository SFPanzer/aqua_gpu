use glam::{Mat4, Quat, Vec3};

pub struct Camera {
    position: Vec3,
    rotation: Quat,
    fov: f32,
    near_plane: f32,
    far_plane: f32,
}

impl Camera {
    pub fn new(position: Vec3, rotation: Quat, fov: f32, near_plane: f32, far_plane: f32) -> Self {
        Self {
            position,
            rotation,
            fov,
            near_plane,
            far_plane,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let dir = self.rotation * Vec3::Z;
        let up = self.rotation * Vec3::Y;
        Mat4::look_to_rh(self.position, dir, up)
    }

    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh_gl(self.fov, aspect_ratio, self.near_plane, self.far_plane)
    }
}
