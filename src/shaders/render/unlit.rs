pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec4 position;
            layout(location = 1) in vec4 velocity;

            layout(location = 0) out float v_speed;

            layout(set = 0, binding = 0) uniform Data {
                mat4 view;
                mat4 proj;
            } uniforms;

            void main() {
                gl_Position = uniforms.proj * uniforms.view * vec4(position.xyz, 1.0);
                v_speed = length(velocity);
                gl_PointSize = 3.0;
            }
        ",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in float v_speed;

            layout(location = 0) out vec4 f_color;

            void main() {
                float max_speed = 3.0;
                float t = clamp(v_speed / max_speed, 0.0, 1.0);

                vec3 color = mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 1.0, 0.0), t);
                f_color = vec4(color, 1.0);
            }
        ",
    }
}
