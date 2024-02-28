#pragma once
#include <glm/gtc/random.hpp> // random sample on hemisphere
#include <algorithm>

struct Cubemap {
    int width, height;
    glm::vec4* data;
};

glm::vec3 sampleCubemapDirection(Cubemap& cubemap, const glm::vec3& direction) {}

glm::vec4 rgbe_to_float(glm::u8vec4 col) {
    // process zero exponent
    if (col == glm::u8vec4(0, 0, 0, 0)) return glm::vec4(0, 0, 0, 0);

    int exp = int(col.a) - 128;
    return glm::vec4(
        std::ldexp((col.r + 0.5f) / 256.0f, exp),
        std::ldexp((col.g + 0.5f) / 256.0f, exp),
        std::ldexp((col.b + 0.5f) / 256.0f, exp),
        1.0f
    );
}

glm::vec4 float_to_rgbe(glm::vec3 col) {
    float d = std::max(col.r, std::max(col.g, col.b));

    //1e-32 is from the radiance code, and is probably larger than strictly necessary:
    if (d <= 1e-32f) {
        return glm::u8vec4(0, 0, 0, 0);
    }

    int e;
    float fac = 255.999f * (std::frexp(d, &e) / d);

    //value is too large to represent, clamp to bright white:
    if (e > 127) {
        return glm::u8vec4(0xff, 0xff, 0xff, 0xff);
    }

    //scale and store:
    return glm::u8vec4(
        std::max(0, int32_t(col.r * fac)),
        std::max(0, int32_t(col.g * fac)),
        std::max(0, int32_t(col.b * fac)),
        e + 128
    );
}

glm::vec3 ndcToDirection(float u, float v) {
    glm::vec3 direction;

    // use max to determine which face is pointing at
    float absU = std::abs(u);
    float absV = std::abs(v);
    float maxComponent = std::max({ absU, absV, 1.0f });

    if (maxComponent == absU) {
        // left or right
        if (u > 0) {
            // right
            direction = glm::vec3(1.0f, -v, -1.0f / u);
        }
        else {
            // left
            direction = glm::vec3(-1.0f, -v, 1.0f / u);
        }
    }
    else if (maxComponent == absV) {
        // up or down
        if (v > 0) {
            // up
            direction = glm::vec3(u, 1.0f, -1.0f / v);
        }
        else {
            // down
            direction = glm::vec3(u, -1.0f, 1.0f / v);
        }
    }
    else {
        // front or back
        if (u > 0) {
            // front
            direction = glm::vec3(u, -v, -1.0f);
        }
        else {
            // back
            direction = glm::vec3(-u, -v, 1.0f);
        }
    }

    // nomorlize direction
    return glm::normalize(direction);
}

glm::vec2 directionToUV(const glm::vec3& dir) {
    float u, v;
    float absX = std::abs(dir.x);
    float absY = std::abs(dir.y);
    float absZ = std::abs(dir.z);
    float maxComponent = std::max({ absX, absY, absZ });
    int faceIndex;

    if (maxComponent == absX) {
        faceIndex = dir.x > 0 ? 0 : 1; // +x or -x
        u = dir.x > 0 ? -dir.z : dir.z;
        v = -dir.y;
    }
    else if (maxComponent == absY) {
        faceIndex = dir.y > 0 ? 2 : 3; // +y or -y 
        u = dir.x;
        v = dir.y > 0 ? dir.z : -dir.z;
    }
    else {
        faceIndex = dir.z > 0 ? 4 : 5; // +z or -z
        u = dir.z > 0 ? dir.x : -dir.x;
        v = -dir.y;
    }

    float faceHeight = 1.0f / 6.0f;
    u = 0.5f * (u / maxComponent + 1.0f); // convert to [0, 1] range
    v = 0.5f * (v / maxComponent + 1.0f) + faceIndex * faceHeight; // convert to the corresponding face

    return glm::vec2(u, v);
}

glm::vec3 randomHemisphereSample(const glm::vec3& normal) {
    glm::vec3 sample = glm::sphericalRand(1.0f);
    if (glm::dot(sample, normal) < 0.0f) {
        sample = -sample;
    }
    return sample;
}