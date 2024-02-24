#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec3 fragPos;
layout(location = 2) in vec3 fragNorm;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D normalMap;
layout(binding = 3) uniform samplerCube envMap;

layout(push_constant) uniform PushConstants {
    mat4 model;
    int materialType;
} pushConstants;

void main() {
    vec3 albedo = fragColor.rgb;
    vec3 normal = fragNorm;
    vec3 viewDir = normalize(ubo.cameraPos - fragPos);
    vec3 reflectDir = reflect(viewDir, normal);

    vec3 envColor = texture(envMap, reflectDir).rgb;

    vec3 color = envColor;

    outColor = vec4(color, 1.0);
}