#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in mat3 fragTBN;
layout(location = 4) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D normalMap;

void main() {
    vec3 normalFromMap = texture(normalMap, fragTexCoord).rgb;
    normalFromMap = normalFromMap * 2.0 - 1.0;
    vec3 normal = normalize(fragTBN * normalFromMap);

    vec3 light = mix(vec3(0,0,0), vec3(1,1,1), dot(normal, vec3(1,0,1)) * 0.5 + 0.5);

    outColor = vec4(light * (fragColor.xyz * texture(texSampler, fragTexCoord).rgb), 1.0);
}