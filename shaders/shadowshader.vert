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
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec4 inColor;


void main() {
    gl_Position = ubo.lightSpace * pushConstants.model * vec4(inPosition, 1.0);
}