#define PI 3.1415926535897932384626433832795

struct Vertex {
    vec3 position;
    vec3 normal;
};

struct GeometryBinding {
    uint64_t vertexAddr;
    uint64_t indexAddr;
    uint materialIdx; 
    uint _pad;
};

struct RayPayload {
    vec4 albedo_hit; // first 3 = albedo, 4th = hit
    vec3 emission;
    vec3 normal;
    vec3 position;
};

// https://arugl.medium.com/hash-noise-in-gpu-shaders-210188ac3a3e
float rand_from_seed(inout uint seed) {
 int k;
 int s = int(seed);
 if (s == 0)
 s = 305420679;
 k = s / 127773;
 s = 16807 * (s - k * 127773) - 2836 * k;
 if (s < 0)
  s += 2147483647;
 seed = uint(s);
 return float(seed % uint(65536)) / 65535.0;
}

float rand_from_seed_m1_p1(inout uint seed) {
 return rand_from_seed(seed) * 2.0 - 1.0;
}

uint hash(uint x) {
 x = ((x >> uint(16)) ^ x) * uint(73244475);
 x = ((x >> uint(16)) ^ x) * uint(73244475);
 x = (x >> uint(16)) ^ x;
 return x;
}

vec3 randomCosineHemisphere(inout uint seed) {
    float sampleX = rand_from_seed(seed);
    float sampleY = rand_from_seed(seed);

    float phi = 2.0 * PI * sampleX;
    float cosTheta = sqrt(sampleY);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float x = cos(phi) * sinTheta;
    float y = sin(phi) * sinTheta;
    float z = cosTheta;

    return vec3(x, y, z);
}
