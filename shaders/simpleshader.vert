#version 450

layout(binding = 0) uniform UniformBufferObject {
     mat4 view;
    mat4 proj;
    mat4 lightSpace;
    mat4 previousModelMat;
    mat4 previousViewMat;
    mat4 previousProjMat;
    vec4 cameraPos;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
	vec4 albedoColor;
    vec2 motionVector;
	float roughness;
	float metalness;
	int materialType;
    int lightCount;
} pushConstants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNorm;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNorm;
layout(location = 2) out vec4 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * pushConstants.model * vec4(inPosition, 1.0);
    fragPos = vec3(pushConstants.model * vec4(inPosition, 1.0));

    fragColor = inColor;
    fragNorm = inNorm;
    
}