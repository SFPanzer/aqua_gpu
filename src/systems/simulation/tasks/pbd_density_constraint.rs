use std::sync::Arc;

use vulkano::{
    buffer::BufferContents, descriptor_set::WriteDescriptorSet, device::Device, shader::EntryPoint,
};

use crate::core::{Aabb, Particles};

use super::compute_task::{ComputeGpuTask, ComputeGpuTaskConstants};

/// PBD密度约束求解常量
/// 实现博客中的Position Based Fluids算法
#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct PbdDensityConstraintConstants {
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    particle_count: u32,
    rest_density: f32,
    smoothing_radius: f32,
    smoothing_radius_sq: f32,
    spiky_kernel_factor: f32,
    spiky_grad_kernel_factor: f32,
    constraint_epsilon: f32,
    relaxation_factor: f32,
    grid_size: f32,
    max_neighbors: u32,
}

impl PbdDensityConstraintConstants {
    pub fn new(
        particle_count: u32,
        rest_density: f32,
        smoothing_radius: f32,
        constraint_epsilon: f32,
        relaxation_factor: f32,
        grid_size: f32,
        aabb: Aabb,
    ) -> Self {
        let smoothing_radius_sq = smoothing_radius * smoothing_radius;

        // Spiky kernel factor for density calculation: 15 / (π * h^6)
        let spiky_kernel_factor = 15.0 / (std::f32::consts::PI * smoothing_radius.powi(6));

        // Spiky gradient kernel factor: -45 / (π * h^6)
        let spiky_grad_kernel_factor = -45.0 / (std::f32::consts::PI * smoothing_radius.powi(6));

        let aabb_min = aabb.min().extend(0.).to_array();
        let aabb_max = aabb.max().extend(0.).to_array();

        Self {
            aabb_min,
            aabb_max,
            particle_count,
            rest_density,
            smoothing_radius,
            smoothing_radius_sq,
            spiky_kernel_factor,
            spiky_grad_kernel_factor,
            constraint_epsilon,
            relaxation_factor,
            grid_size,
            max_neighbors: 96, // 参考博客中的设置
        }
    }
}

/// 计算拉格朗日乘子
impl ComputeGpuTaskConstants for PbdDensityConstraintConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/pbd_calc_lambda.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.predicted_position().clone()), // predicted_positions
            WriteDescriptorSet::buffer(1, particles.density().clone()),            // densities
            WriteDescriptorSet::buffer(2, particles.contacts().clone()),           // contacts
            WriteDescriptorSet::buffer(3, particles.contact_counts().clone()),     // contact_counts
            WriteDescriptorSet::buffer(4, particles.lambda().clone()),             // lambdas
        ]
    }

    fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

pub(crate) type PbdCalcLambdaTask = ComputeGpuTask<PbdDensityConstraintConstants>;

/// 计算位移
#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct PbdCalcDisplacementConstants {
    inner: PbdDensityConstraintConstants,
}

impl PbdCalcDisplacementConstants {
    pub fn new(constants: PbdDensityConstraintConstants) -> Self {
        Self { inner: constants }
    }
}

impl ComputeGpuTaskConstants for PbdCalcDisplacementConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/pbd_calc_displacement.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.predicted_position().clone()), // predicted_positions
            WriteDescriptorSet::buffer(1, particles.lambda().clone()),             // lambdas
            WriteDescriptorSet::buffer(2, particles.contacts().clone()),           // contacts
            WriteDescriptorSet::buffer(3, particles.contact_counts().clone()),     // contact_counts
            WriteDescriptorSet::buffer(4, particles.delta_position().clone()), // delta_positions
        ]
    }

    fn particle_count(&self) -> u32 {
        self.inner.particle_count
    }
}

pub(crate) type PbdCalcDisplacementTask = ComputeGpuTask<PbdCalcDisplacementConstants>;

/// 应用位移
#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct PbdApplyDisplacementConstants {
    inner: PbdDensityConstraintConstants,
}

impl PbdApplyDisplacementConstants {
    pub fn new(constants: PbdDensityConstraintConstants) -> Self {
        Self { inner: constants }
    }
}

impl ComputeGpuTaskConstants for PbdApplyDisplacementConstants {
    fn entry_point(device: &Arc<Device>) -> EntryPoint {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/simulation/pbd_apply_displacement.comp",
            }
        }
        cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap()
    }

    fn descriptor_writes(particles: &Particles) -> impl IntoIterator<Item = WriteDescriptorSet> {
        [
            WriteDescriptorSet::buffer(0, particles.predicted_position().clone()), // predicted_positions (read/write)
            WriteDescriptorSet::buffer(1, particles.delta_position().clone()), // delta_positions (read)
        ]
    }

    fn particle_count(&self) -> u32 {
        self.inner.particle_count
    }
}

pub(crate) type PbdApplyDisplacementTask = ComputeGpuTask<PbdApplyDisplacementConstants>;

/// 计算拉格朗日乘子 / 计算位移 / 应用位移
pub struct PbdDensityConstraintTask {
    calc_lambda: PbdCalcLambdaTask,
    calc_displacement: PbdCalcDisplacementTask,
    apply_displacement: PbdApplyDisplacementTask,
}

impl PbdDensityConstraintTask {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            calc_lambda: PbdCalcLambdaTask::new(device),
            calc_displacement: PbdCalcDisplacementTask::new(device),
            apply_displacement: PbdApplyDisplacementTask::new(device),
        }
    }

    pub fn set_constants(&mut self, constants: PbdDensityConstraintConstants) {
        self.calc_lambda.set_constants(constants);
        self.calc_displacement
            .set_constants(PbdCalcDisplacementConstants::new(constants));
        self.apply_displacement
            .set_constants(PbdApplyDisplacementConstants::new(constants));
    }

    pub fn update_descriptor_set(
        &mut self,
        descriptor_set_allocator: &Arc<
            vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator,
        >,
        particles: &mut Particles,
    ) {
        self.calc_lambda
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.calc_displacement
            .update_descriptor_set(descriptor_set_allocator, particles);
        self.apply_displacement
            .update_descriptor_set(descriptor_set_allocator, particles);
    }

    pub fn execute_iteration(&mut self, executor: &impl crate::utils::GpuTaskExecutor) {
        // 计算拉格朗日乘子
        executor.execute(&mut self.calc_lambda);

        // 计算位移
        executor.execute(&mut self.calc_displacement);

        // 应用位移到预测位置
        executor.execute(&mut self.apply_displacement);
    }
}