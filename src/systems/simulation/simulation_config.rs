use glam::Vec3;

use crate::core::Aabb;

#[derive(Clone, Debug)]
pub(crate) struct SimulationConfig {
    // Basic simulation parameters
    pub simulation_aabb: Aabb,
    pub gravity: Vec3,

    // Time step limits (for numerical stability)
    pub max_time_step: f32,
    pub min_time_step: f32,

    // Spatial partitioning parameters
    pub grid_size: f32,

    // SPH fluid simulation parameters
    pub sph_params: SphParams,

    // Performance optimization parameters
    #[allow(dead_code)]
    pub max_neighbors: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct SphParams {
    /// Particle mass (kg)
    pub particle_mass: f32,
    /// Kernel smoothing radius (m)
    pub smoothing_radius: f32,
    /// Rest density (kg/m³)
    #[allow(dead_code)]
    pub rest_density: f32,
    /// Viscosity coefficient
    #[allow(dead_code)]
    pub viscosity: f32,
    /// Surface tension coefficient
    #[allow(dead_code)]
    pub surface_tension: f32,

    // PBD specific parameters
    /// Number of PBD solver iterations
    pub pbd_iterations: u32,
    /// Epsilon for PBD density constraint (to prevent division by zero and stabilize)
    pub pbd_constraint_epsilon: f32,
    /// Relaxation factor for PBD position correction (typically between 0.1 and 1.0)
    pub pbd_relaxation_factor: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        let sph_params = SphParams::default();

        Self {
            simulation_aabb: Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            gravity: Vec3::new(0.0, -9.81, 0.0),

            // Time step limits - ensure numerical stability
            max_time_step: 1.0 / 60.0, // Maximum 16ms, smaller steps for stability
            min_time_step: 1.0 / 300.0, // Minimum 3ms, prevent too small time steps

            // grid_size should be around 0.5-1.0 times smoothing_radius for balance between accuracy and performance
            grid_size: sph_params.smoothing_radius * 0.8,

            sph_params,
            max_neighbors: 32,
        }
    }
}

impl Default for SphParams {
    fn default() -> Self {
        Self {
            particle_mass: 0.001,   // 1g per particle - lighter for stability
            smoothing_radius: 0.1,  // Smaller radius for 1g particles
            rest_density: 10.0,     // Much lower rest density for 1g particles in small volume
            viscosity: 0.001,       // Water viscosity
            surface_tension: 0.073, // Water surface tension

            // Stronger PBD parameters for better stacking
            pbd_iterations: 3, // More iterations for stronger constraints
            pbd_constraint_epsilon: 1e-4, // Tighter epsilon for better precision
            pbd_relaxation_factor: 0.8, // Higher relaxation for stronger effect
        }
    }
}

impl SimulationConfig {
    /// Create high performance configuration (fewer particles, high framerate)
    #[allow(dead_code)]
    pub fn high_performance() -> Self {
        Self {
            max_time_step: 1.0 / 60.0,  // Allow higher framerate
            min_time_step: 1.0 / 300.0, // More precise time control
            sph_params: SphParams {
                smoothing_radius: 0.15,
                ..SphParams::default()
            },
            grid_size: 0.15 * 0.8,
            max_neighbors: 32, // Reduce neighborhood particle count
            ..Self::default()
        }
    }

    /// Create high quality configuration (more particles, lower framerate)
    #[allow(dead_code)]
    pub fn high_quality() -> Self {
        Self {
            max_time_step: 1.0 / 20.0, // Allow lower framerate but maintain stability
            min_time_step: 1.0 / 120.0,
            sph_params: SphParams {
                smoothing_radius: 0.25,
                ..SphParams::default()
            },
            grid_size: 0.25 * 0.7,
            max_neighbors: 128, // Increase neighborhood particle count for precision
            ..Self::default()
        }
    }

    /// Create large scale simulation configuration (million particle level)
    #[allow(dead_code)]
    pub fn large_scale() -> Self {
        Self {
            simulation_aabb: Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0)),
            max_time_step: 1.0 / 20.0, // Large scale simulations typically have lower framerate
            min_time_step: 1.0 / 120.0,
            sph_params: SphParams {
                smoothing_radius: 0.1, // Smaller radius to support more particles
                ..SphParams::default()
            },
            grid_size: 0.1 * 0.8,
            max_neighbors: 64,
            ..Self::default()
        }
    }

    /// Create configuration optimized for stable stacking
    #[allow(dead_code)]
    pub fn stable_stacking() -> Self {
        Self {
            simulation_aabb: Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            gravity: Vec3::new(0.0, -5.0, 0.0), // Reduced gravity for better control
            max_time_step: 1.0 / 120.0,         // Very small time steps
            min_time_step: 1.0 / 300.0,
            sph_params: SphParams {
                particle_mass: 0.001,
                smoothing_radius: 0.08, // Smaller radius for tighter packing
                rest_density: 15.0,     // Slightly higher for more constraint force
                viscosity: 0.01,        // Higher viscosity for damping
                surface_tension: 0.073,
                pbd_iterations: 5,            // More iterations for stability
                pbd_constraint_epsilon: 1e-5, // Very tight constraint
                pbd_relaxation_factor: 0.6,   // Moderate relaxation
            },
            grid_size: 0.08 * 0.9,
            max_neighbors: 64,
        }
    }

    /// Create configuration for fountain spray effect
    #[allow(dead_code)]
    pub fn fountain_spray() -> Self {
        Self {
            simulation_aabb: Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
            gravity: Vec3::new(0.01, -9.8, 0.01),
            max_time_step: 1.0 / 60.0,
            min_time_step: 1.0 / 240.0,
            sph_params: SphParams {
                particle_mass: 0.001,
                smoothing_radius: 0.03,
                rest_density: 25.0,
                viscosity: 0.005,
                surface_tension: 0.073,
                pbd_iterations: 6,
                pbd_constraint_epsilon: 1e-5,
                pbd_relaxation_factor: 0.8,
            },
            grid_size: 0.02 * 0.5,
            max_neighbors: 128,
        }
    }

    /// Create configuration for tube/cylinder container stacking test
    #[allow(dead_code)]
    pub fn tube_stacking() -> Self {
        Self {
            // 创建一个管状容器：细长的圆柱形
            simulation_aabb: Aabb::new(
                Vec3::new(-0.2, -1.0, -0.2), // 小的横截面（0.4m直径）
                Vec3::new(0.2, 1.0, 0.2),    // 高2m的管子
            ),
            gravity: Vec3::new(0.0, -9.81, 0.0), // 标准重力
            max_time_step: 1.0 / 120.0,          // 小时间步长确保稳定性
            min_time_step: 1.0 / 300.0,
            sph_params: SphParams {
                particle_mass: 0.002,   // 稍微重一点的粒子
                smoothing_radius: 0.06, // 较小的影响半径适合密集堆叠
                rest_density: 20.0,     // 较高的静息密度促进堆叠
                viscosity: 0.02,        // 增加粘性帮助稳定
                surface_tension: 0.073,
                pbd_iterations: 8,            // 更多迭代确保约束满足
                pbd_constraint_epsilon: 1e-6, // 非常严格的约束
                pbd_relaxation_factor: 0.5,   // 温和的松弛因子
            },
            grid_size: 0.06 * 0.8, // 网格大小略小于影响半径
            max_neighbors: 96,     // 增加邻居数以处理密集堆叠
        }
    }

    /// Clamp time step within reasonable range
    pub fn clamp_time_step(&self, dt: f32) -> f32 {
        dt.clamp(self.min_time_step, self.max_time_step)
    }

    /// Validate configuration parameter reasonableness
    #[allow(dead_code)]
    pub fn validate(&self) -> Result<(), String> {
        if self.grid_size <= 0.0 {
            return Err("grid_size must be greater than 0".to_string());
        }

        if self.sph_params.smoothing_radius <= 0.0 {
            return Err("smoothing_radius must be greater than 0".to_string());
        }

        if self.grid_size > self.sph_params.smoothing_radius {
            return Err(format!(
                "grid_size ({}) should not be greater than smoothing_radius ({}) to ensure neighborhood search efficiency",
                self.grid_size, self.sph_params.smoothing_radius
            ));
        }

        if self.min_time_step <= 0.0 || self.max_time_step <= 0.0 {
            return Err("Time step limits must be greater than 0".to_string());
        }

        if self.min_time_step >= self.max_time_step {
            return Err("min_time_step must be less than max_time_step".to_string());
        }

        Ok(())
    }

    /// Print configuration information
    #[allow(dead_code)]
    pub fn print_info(&self) {
        println!("=== Simulation Configuration ===");
        println!("Simulation space: {:?}", self.simulation_aabb);
        println!("Gravity: {:?}", self.gravity);
        println!(
            "Time step limits: {:.6}s - {:.6}s",
            self.min_time_step, self.max_time_step
        );
        println!("Grid size: {:.4}m", self.grid_size);
        println!(
            "SPH kernel radius: {:.4}m",
            self.sph_params.smoothing_radius
        );
        println!("Particle mass: {:.4}kg", self.sph_params.particle_mass);
        println!("Max neighbors: {}", self.max_neighbors);
        println!(
            "Grid size / kernel radius ratio: {:.2}",
            self.grid_size / self.sph_params.smoothing_radius
        );
        println!("================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = SimulationConfig::default();
        assert!(
            config.validate().is_ok(),
            "Default configuration should be valid"
        );

        // Verify default parameter reasonableness
        assert!(config.grid_size <= config.sph_params.smoothing_radius);
        assert!(config.grid_size > 0.0);
        assert!(config.sph_params.smoothing_radius > 0.0);
        assert!(config.min_time_step > 0.0);
        assert!(config.max_time_step > 0.0);
        assert!(config.min_time_step < config.max_time_step);
    }

    #[test]
    fn test_preset_configs_validation() {
        let configs = vec![
            ("high_performance", SimulationConfig::high_performance()),
            ("high_quality", SimulationConfig::high_quality()),
            ("large_scale", SimulationConfig::large_scale()),
        ];

        for (name, config) in configs {
            assert!(
                config.validate().is_ok(),
                "{} configuration should be valid",
                name
            );

            // Verify grid_size to smoothing_radius relationship
            let ratio = config.grid_size / config.sph_params.smoothing_radius;
            assert!(
                ratio >= 0.5 && ratio <= 1.0,
                "{} config grid size/kernel radius ratio ({:.2}) should be in [0.5, 1.0] range",
                name,
                ratio
            );
        }
    }

    #[test]
    fn test_config_info_display() {
        let config = SimulationConfig::large_scale();
        println!("\nLarge scale simulation configuration:");
        config.print_info();

        // Verify key parameters
        assert_eq!(config.sph_params.smoothing_radius, 0.1);
        assert!((config.grid_size - 0.08).abs() < 1e-6); // 0.1 * 0.8, allow floating point error
        assert_eq!(config.max_neighbors, 64);
    }

    #[test]
    fn test_grid_size_optimization() {
        let config = SimulationConfig::default();

        // Verify grid_size is optimized for kernel radius
        println!("\nDefault configuration kernel and grid parameters:");
        println!("Kernel radius: {}m", config.sph_params.smoothing_radius);
        println!("Grid size: {}m", config.grid_size);
        println!(
            "Ratio: {:.2}",
            config.grid_size / config.sph_params.smoothing_radius
        );

        // This ratio should be in reasonable range to balance accuracy and performance
        let ratio = config.grid_size / config.sph_params.smoothing_radius;
        assert!(
            ratio >= 0.6 && ratio <= 0.9,
            "Grid size should be about 60%-90% of kernel radius, actual ratio: {:.2}",
            ratio
        );
    }

    #[test]
    fn test_time_step_clamping() {
        let config = SimulationConfig::default();

        println!("\nTime step limit test:");
        println!("Min time step: {:.6}s", config.min_time_step);
        println!("Max time step: {:.6}s", config.max_time_step);

        // Test time step limiting functionality
        assert_eq!(config.clamp_time_step(0.001), config.min_time_step); // Too small
        assert_eq!(config.clamp_time_step(0.1), config.max_time_step); // Too large
        assert_eq!(config.clamp_time_step(0.01), 0.01); // Normal range - within default limits

        // Verify actual game scenario time steps
        let dt_60fps = 1.0 / 60.0; // ~16.67ms
        let dt_30fps = 1.0 / 30.0; // ~33.33ms
        let dt_120fps = 1.0 / 120.0; // ~8.33ms

        assert_eq!(config.clamp_time_step(dt_60fps), dt_60fps);
        // 30 FPS时间步长超过max_time_step，会被钳制
        assert_eq!(config.clamp_time_step(dt_30fps), config.max_time_step);
        assert_eq!(config.clamp_time_step(dt_120fps), dt_120fps);

        println!(
            "60 FPS time step ({:.6}s) -> {:.6}s",
            dt_60fps,
            config.clamp_time_step(dt_60fps)
        );
        println!(
            "30 FPS time step ({:.6}s) -> {:.6}s",
            dt_30fps,
            config.clamp_time_step(dt_30fps)
        );
        println!(
            "120 FPS time step ({:.6}s) -> {:.6}s",
            dt_120fps,
            config.clamp_time_step(dt_120fps)
        );
    }

    #[test]
    fn test_dynamic_time_step_behavior() {
        println!("\nDynamic time step behavior test:");

        // Simulate different frame time scenarios
        let scenarios = vec![
            ("Stable 60 FPS", 1.0 / 60.0),
            ("Stable 30 FPS", 1.0 / 30.0),
            ("High framerate 120 FPS", 1.0 / 120.0),
            ("Stutter case", 0.1),               // 100ms stutter
            ("Extremely high framerate", 0.001), // 1000 FPS
        ];

        let config = SimulationConfig::default();

        for (scenario, raw_dt) in scenarios {
            let clamped_dt = config.clamp_time_step(raw_dt);
            let effective_fps = 1.0 / clamped_dt;

            println!(
                "{}: raw dt={:.6}s -> clamped dt={:.6}s (effective framerate: {:.1} FPS)",
                scenario, raw_dt, clamped_dt, effective_fps
            );

            // Verify all time steps are within reasonable range
            assert!(clamped_dt >= config.min_time_step);
            assert!(clamped_dt <= config.max_time_step);
        }
    }
}
