#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D normalMap;
layout(binding = 3) uniform samplerCube envMap;
layout(binding = 4) uniform samplerCube lambertianMap;

layout(push_constant) uniform PushConstants {
    mat4 model;
	vec4 albedoColor;
	float roughness;
	float metalness;
	int materialType;
} pushConstants;


void main() {
    vec3 normal = normalize(fragNorm);
    vec3 lightDir = normalize(vec3(1,1,1));

    float diff = dot(normal, lightDir) * 0.5 + 0.5;
    vec3 finalColor = diff * fragColor.rgb;

    outColor = vec4(finalColor, 1.0);
}

