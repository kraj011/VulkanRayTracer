#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_image_load_formatted : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    payload.albedo_hit = vec4(0.0);
    payload.emission = vec3(0.0);
    payload.normal = vec3(0.0);
    payload.position = vec3(0.0);
}