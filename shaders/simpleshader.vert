#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
	vec4 albedoColor;
	float roughness;
	float metalness;
	int materialType;
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