#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in mat3 fragTBN;
layout(location = 4) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D normalMap;
layout(binding = 3) uniform samplerCube envMap;

layout(push_constant) uniform PushConstants {
    mat4 model;
    int materialType;
} pushConstants;

void main() {
    vec3 albedo;
    vec3 normal;

    if (pushConstants.materialType == 1) {
        albedo = texture(texSampler, fragTexCoord).rgb;
        vec3 normalFromMap = texture(normalMap, fragTexCoord).rgb;
        normal = normalize(fragTBN * (normalFromMap * 2.0 - 1.0));
    } else if (pushConstants.materialType == 0) {
        albedo = fragColor.rgb;
        normal = normalize(fragTBN[2]);
    }

    vec3 reflectedRay = reflect(-normalize(fragColor.rgb), normal);
    vec3 envColor = texture(envMap, reflectedRay).rgb;

    vec3 light = mix(vec3(0, 0, 0), vec3(1, 1, 1), dot(normal, vec3(1, 0, 1)) * 0.5 + 0.5);
    vec3 color = light * (albedo + envColor);

    outColor = vec4(color, 1.0);
}