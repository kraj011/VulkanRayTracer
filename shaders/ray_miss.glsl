#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    payload.albedo_hit = vec4(0.0);
    payload.emission = vec3(0.0);
    payload.normal = vec3(0.0);
    payload.position = vec3(0.0);
}