#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
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
layout(location = 5) out vec3 fragPos;
layout(location = 6) out vec3 fragNorm;

void main() {
    gl_Position = ubo.proj * ubo.view * pushConstants.model * vec4(inPosition, 1.0);
    fragPos = vec3(pushConstants.model * vec4(inPosition, 1.0));

    if (pushConstants.materialType == 1) {
    vec3 T = normalize((pushConstants.model * vec4(inTangent.xyz, 0.0)).xyz);
    vec3 N = normalize((pushConstants.model * vec4(inNorm, 0.0)).xyz);
    vec3 B = cross(N, T) * inTangent.w;

    fragTBN = mat3(T, B, N);
    } else {
    fragTBN = mat3(1.0);
    }
    

    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fragNorm = inNorm;
    
}