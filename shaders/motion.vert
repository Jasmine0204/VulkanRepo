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

layout(location = 0) out vec2 motionVector;


void main() {
    vec4 currentPos = ubo.proj * ubo.view * pushConstants.model * vec4(inPosition, 1.0);
    vec4 previousPos = ubo.previousProjMat * ubo.previousViewMat * ubo.previousModelMat * vec4(inPosition, 1.0);

    motionVector = (currentPos.xy / currentPos.w - previousPos.xy / previousPos.w);

    if (length(motionVector) < 0.01) {
        motionVector = vec2(0.0);
    }

    gl_Position = currentPos;
}