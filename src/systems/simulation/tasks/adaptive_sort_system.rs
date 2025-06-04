use std::sync::Arc;
use vulkano::{descriptor_set::allocator::StandardDescriptorSetAllocator, device::Device};

use crate::{core::Particles, utils::GpuTaskExecutor};

use super::radix_sort_system::RadixSortSystem;

#[allow(dead_code)]
pub struct AdaptiveSortSystem {
    sort_system: RadixSortSystem,
    last_sort_frame: u32,
    sort_interval: u32,      // Frames between sorts
    movement_threshold: f32, // Threshold for particle movement
}

impl AdaptiveSortSystem {
    #[allow(unused)]
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            sort_system: RadixSortSystem::new(device),
            last_sort_frame: 0,
            sort_interval: 4,        // Sort every 4 frames by default
            movement_threshold: 0.1, // Sort when particles move > 10% of cell size
        }
    }

    /// Conditionally sort particles based on movement and time
    #[allow(unused)]
    pub fn update_sort(
        &mut self,
        particles: &mut Particles,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        executor: &impl GpuTaskExecutor,
        current_frame: u32,
        force_sort: bool,
    ) -> bool {
        let should_sort =
            force_sort || (current_frame - self.last_sort_frame >= self.sort_interval);

        if should_sort {
            self.sort_system
                .sort_morton_codes(particles, descriptor_set_allocator, executor);
            self.last_sort_frame = current_frame;
            true
        } else {
            false
        }
    }

    /// Set the interval between sorts (in frames)  
    #[allow(unused)]
    pub fn set_sort_interval(&mut self, interval: u32) {
        self.sort_interval = interval;
    }

    /// Get current sort interval
    #[allow(unused)]
    pub fn sort_interval(&self) -> u32 {
        self.sort_interval
    }
}
