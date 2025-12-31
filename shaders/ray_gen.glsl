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