pub mod preiew_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec3 position;
           
            layout(push_constant) uniform PushConstants {
                mat4 mvp;
                vec3 col;
            } pc;

            void main() {
                vec4 world_pos = pc.mvp * vec4(position, 1.f);
                gl_Position = world_pos;
            }
        "
    }
}

pub mod preiew_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            layout(push_constant) uniform PushConstants {
                mat4 mvp;
                vec3 col;
            } pc;

            void main() {
                f_color = vec4(pc.col, 1.0);
            }
        "
    }
}
