#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in mat3 fragTBN;
layout(location = 4) in vec2 fragTexCoord;
layout(location = 5) in vec3 fragPos;
layout(location = 6) in vec3 fragNorm;

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

    if (pushConstants.materialType == 1) {
        albedo = texture(texSampler, fragTexCoord).rgb;
        vec3 normalFromMap = texture(normalMap, fragTexCoord).rgb;
        normal = normalize(fragTBN * (normalFromMap * 2.0 - 1.0));
    }

    vec3 envColor = texture(envMap, reflectDir).rgb;

    vec3 color = envColor;

    outColor = vec4(color, 1.0);
}