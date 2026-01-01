#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_image_load_formatted : enable

#include "common.glsl"

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 1, set = 0) uniform image2D image;

layout(push_constant) uniform PushConstants {
    mat4 viewInv;
    mat4 projInv;
} pc;

layout(location = 0) rayPayloadEXT RayPayload payload;

const int MAX_BOUNCES = 5;

void main() {
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;
    
    vec4 origin = pc.viewInv * vec4(0.0, 0.0, 0.0, 1.0);

    vec4 target = pc.projInv * vec4(d, 1.0, 1.0);
    vec4 dir = pc.viewInv * vec4(normalize(target.xyz), 0.0);

    float tmin = 0.001;
    float tmax = 10000.0;

    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);

    uint seed = hash(gl_LaunchIDEXT.x * 1920);
    
    for(int i = 0; i < MAX_BOUNCES; i++) {
        payload.albedo_hit = vec4(0.0);

        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, dir.xyz, tmax, 0);
        
        // no hit
        if(payload.albedo_hit.w < 1e-3) break;
        
        radiance += throughput * payload.emission;

        vec3 out_dir = normalize(randomCosineHemisphere(seed));
        vec3 brdf = payload.albedo_hit.rgb * (1.0 / PI);

        float pdf = max(0.0, dot(out_dir, payload.normal)) / PI;
        throughput *= brdf * max(0.0, dot(payload.normal, out_dir)) / max(pdf, 1e-8);

        origin = vec4(payload.position, 1.0);
        dir = vec4(out_dir, 0.0);

        if(max(throughput.r, max(throughput.g, throughput.b)) < 1e-3) break;
    }

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(radiance, 1.0));
}