#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require


#include "common.glsl"

// https://docs.vulkan.org/samples/latest/samples/extensions/buffer_device_address/README.html
layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex verts[];
};

layout(buffer_reference, scalar) readonly buffer IndexBuffer {
    uint idxs[];
};

layout(binding = 2, set = 0, scalar) readonly buffer GeometryBindingsBuffer {
    GeometryBinding bindings[];
} geometryBindings;

layout(location = 0) rayPayloadInEXT RayPayload payload;
// https://www.gsn-lib.org/docs/nodes/raytracing.php#:~:text=The%20closest%2Dhit%20and%20miss,as%20possible%20for%20best%20performance.
hitAttributeEXT vec2 baryCoord;

void main() {
    GeometryBinding binding = geometryBindings.bindings[gl_InstanceCustomIndexEXT];
    VertexBuffer vBuff = VertexBuffer(binding.vertexAddr);
    IndexBuffer iBuff = IndexBuffer(binding.indexAddr);

    uint tri_0 = iBuff.idxs[gl_PrimitiveID * 3 + 0];
    uint tri_1 = iBuff.idxs[gl_PrimitiveID * 3 + 1];
    uint tri_2 = iBuff.idxs[gl_PrimitiveID * 3 + 2];

    vec3 normal_0 = vBuff.verts[tri_0].normal;
    vec3 normal_1 = vBuff.verts[tri_1].normal;
    vec3 normal_2 = vBuff.verts[tri_2].normal;

    vec3 bary = vec3(1.0 - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
    vec3 shadingNormal = normalize(normal_0 * bary.x + normal_1 * bary.y + normal_2 * bary.z);
    shadingNormal = normalize(mat3(gl_ObjectToWorldEXT) * shadingNormal);


    vec3 hitPosWorld = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.albedo_hit = vec4(shadingNormal * 0.5 + 0.5, 1.0);;
    payload.normal = shadingNormal;
    payload.emission = vec3(1.0);
    payload.position = hitPosWorld;
}