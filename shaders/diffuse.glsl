#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_image_load_formatted : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    vec3 hitPosWorld = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.albedo_hit = vec4(normalize(hitPosWorld), 1.0);
    // payload.normal = 
    // payload.emission = 
    // payload.position = hitPosWorld;
}