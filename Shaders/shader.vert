#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
    int materialType;
} pushConstants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNorm;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec4 inColor;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out mat3 fragTBN;
layout(location = 4) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * pushConstants.model * vec4(inPosition, 1.0);

    vec3 T = normalize((pushConstants.model * vec4(inTangent.xyz, 0.0)).xyz);
    vec3 N = normalize((pushConstants.model * vec4(inNorm, 0.0)).xyz);

    vec3 B = cross(N, T) * inTangent.w;

    fragTBN = mat3(T, B, N);

    fragColor = inColor;
    fragTexCoord = inTexCoord;
}