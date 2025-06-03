use glam::Vec3;

#[derive(Clone, Copy, Debug, Default)]
pub struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[allow(dead_code)]
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    #[allow(dead_code)]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn min(&self) -> Vec3 {
        self.min
    }

    pub fn max(&self) -> Vec3 {
        self.max
    }
}
