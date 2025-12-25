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
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require
            #extension GL_EXT_shader_image_load_formatted : enable
            
            layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;
            layout(binding = 1, set = 0) uniform image2D image;

            layout(push_constant) uniform PushConstants {
                mat4 viewInv;
                mat4 projInv;
            } pc;

            layout(location = 0) rayPayloadEXT vec4 hitValue;

            void main() {
                const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
                const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
                vec2 d = inUV * 2.0 - 1.0;
                
                vec4 origin = pc.viewInv * vec4(0.0, 0.0, 0.0, 1.0);
                vec4 target = pc.projInv * vec4(d, 1.0, 1.0);
                vec4 dir = pc.viewInv * vec4(normalize(target.xyz), 0.0);

                float tmin = 0.001;
                float tmax = 10000.0;

                hitValue = vec4(0.0);

                traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, dir.xyz, tmax, 0);

                imageStore(image, ivec2(gl_LaunchIDEXT.xy), hitValue);
            }
        "
    }
}

// https://www.khronos.org/blog/ray-tracing-in-vulkan
pub mod closest_hit_shader {
    vulkano_shaders::shader! {
        ty: "closesthit",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require
            #extension GL_EXT_shader_image_load_formatted : enable

            layout(location = 0) rayPayloadInEXT vec4 payload;

            void main() {
                vec3 hitPosWorld = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
                payload = vec4(normalize(hitPosWorld), 1.0);
            }
        "
    }
}

pub mod miss_shader {
    vulkano_shaders::shader! {
        ty: "miss",
        spirv_version: "1.4",
        src: r"
            #version 460
            #extension GL_EXT_ray_tracing : require
            #extension GL_EXT_shader_image_load_formatted : enable

            layout(location = 0) rayPayloadInEXT vec4 payload;

            void main() {
                payload = vec4(0.2, 0.3, 0.5, 1.0);
            }
        "
    }
}
