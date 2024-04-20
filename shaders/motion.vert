#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
    mat4 previousModel;
    vec3 cameraPos;
} ubo;


layout(push_constant) uniform PushConstants {
    mat4 model;
	vec4 albedoColor;
	float roughness;
	float metalness;
	int materialType;
    int lightCount;
} pushConstants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNorm;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec4 inColor;

layout(location = 0) out vec4 fragPosition;
layout(location = 1) out vec3 motionVector;


void main() {
    vec4 currentPos = ubo.proj * ubo.view * pushConstants.model * vec4(inPosition, 1.0);
    vec4 previousPos = ubo.proj * ubo.view * ubo.previousModel * vec4(inPosition, 1.0);

    fragPosition = currentPos;

    motionVector = (currentPos - previousPos).xyz;

    gl_Position = currentPos;
}