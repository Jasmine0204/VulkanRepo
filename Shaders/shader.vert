#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant, std430) uniform pc{
    mat4 model;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNorm;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec3 fragNorm;

void main() {
    gl_Position = ubo.proj * ubo.view * model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragNorm = (model * vec4(inNorm, 0.0)).xyz;
}