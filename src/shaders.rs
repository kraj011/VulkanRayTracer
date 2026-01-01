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

// https://github.com/SaschaWillems/Vulkan/blob/master/shaders/glsl/raytracingbasic/raygen.rgen
pub mod ray_gen_shader {
    vulkano_shaders::shader! {
        ty: "raygen",
        spirv_version: "1.4",
        path: "./shaders/ray_gen.glsl",
        include: ["./shaders"]
    }
}

// https://www.khronos.org/blog/ray-tracing-in-vulkan
pub mod closest_hit_shader {
    vulkano_shaders::shader! {
        ty: "closesthit",
        spirv_version: "1.4",
        path: "./shaders/diffuse.glsl",
        include: ["./shaders"]
    }
}

pub mod miss_shader {
    vulkano_shaders::shader! {
        ty: "miss",
        spirv_version: "1.4",
        path: "./shaders/ray_miss.glsl",
        include: ["./shaders"]
    }
}
