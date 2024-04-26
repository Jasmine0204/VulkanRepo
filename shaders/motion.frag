#version 450

layout(location = 0) in vec2 motionVector;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 normalizedMotion = motionVector * 0.5 + 0.5;
    outColor = vec4(normalizedMotion, 0.0 , 1.0);
}