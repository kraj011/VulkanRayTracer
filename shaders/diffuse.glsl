#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_image_load_formatted : enable

layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
    vec3 hitPosWorld = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload = vec4(normalize(hitPosWorld), 1.0);
}