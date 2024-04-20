#version 450

layout(location = 0) in vec4 fragPosition;
layout(location = 1) in vec3 motionVector;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normalizedMotion = motionVector * 0.5 + 0.5;
    outColor = vec4(normalizedMotion, 1.0);
}