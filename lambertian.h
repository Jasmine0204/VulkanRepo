#pragma once
#include <glm/gtc/random.hpp> // random sample on hemisphere
#include <algorithm>

struct Cubemap {
    int width, height;
    glm::vec4* data;
};

void directionToUV(const glm::vec3& dir, int& faceIndex, float& u, float& v) {
    float absX = std::abs(dir.x);
    float absY = std::abs(dir.y);
    float absZ = std::abs(dir.z);
    float maxComponent = std::max({ absX, absY, absZ });

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

    u = 0.5f * (u / maxComponent + 1.0f);
    v = 0.5f * (v / maxComponent + 1.0f);

}

glm::vec3 sampleCubemapDirection(Cubemap& cubemap, const glm::vec3& direction) {
    int faceIndex;
    float u, v;

    directionToUV(direction, faceIndex, u, v);

    int faceWidth = cubemap.width;
    int faceHeight = cubemap.height / 6;

    int x = static_cast<int>(u * faceWidth);
    int y = static_cast<int>((faceIndex * faceHeight) + v * faceHeight);

    // get color from data 
    int index = y * cubemap.width + x;
    glm::vec4 colorVec;
    if (index >= 0 && index < (cubemap.width * cubemap.height)) {
        colorVec = cubemap.data[index];
    }
    else {
        std::cerr << "Index out of range: " << index << std::endl;
    }

    // turn to vec3 
    glm::vec3 color(colorVec.r, colorVec.g, colorVec.b);

    return color;
}

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

glm::vec3 faceIndexToDirection(int faceIndex, float u, float v) {
    // map uv from [0, 1] to [-1, 1]
    float x = 2.0f * u - 1.0f;
    float y = 2.0f * v - 1.0f;

    glm::vec3 direction;
    switch (faceIndex) {
    case 0: // +X
        direction = glm::vec3(1.0f, -y, -x);
        break;
    case 1: // -X
        direction = glm::vec3(-1.0f, -y, x);
        break;
    case 2: // +Y
        direction = glm::vec3(x, 1.0f, y);
        break;
    case 3: // -Y
        direction = glm::vec3(x, -1.0f, -y);
        break;
    case 4: // +Z
        direction = glm::vec3(x, -y, 1.0f);
        break;
    case 5: // -Z
        direction = glm::vec3(-x, -y, -1.0f);
        break;
    default:
        direction = glm::vec3(0.0f);
        break;
    }

    return glm::normalize(direction);
}

glm::vec3 randomHemisphereSample(const glm::vec3& normal) {
    glm::vec3 sample = glm::sphericalRand(1.0f);
    if (glm::dot(sample, normal) < 0.0f) {
        sample = -sample;
    }
    return sample;
}