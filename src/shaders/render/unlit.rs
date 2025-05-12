pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec3 position;

            layout(set = 0, binding = 0) uniform Data {
                mat4 view;
                mat4 proj;
            } uniforms;

            void main() {
                gl_Position = uniforms.proj * uniforms.view * vec4(position, 1.0);
                gl_PointSize = 10.0;
            }
        ",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
