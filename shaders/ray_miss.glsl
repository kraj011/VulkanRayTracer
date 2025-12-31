#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_image_load_formatted : enable

layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
    payload = vec4(0.2, 0.3, 0.5, 1.0);
}