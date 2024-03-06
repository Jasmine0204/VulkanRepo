#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in mat3 fragTBN;
layout(location = 5) in vec2 fragTexCoord;
layout(location = 6) in vec4 fragColor;

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
    int materialType;
} pushConstants;

vec3 toneMappingFilmic(vec3 color) {
    color = max(vec3(0.0), color - vec3(0.004));
    color = (color * (6.2 * color + vec3(0.5))) / (color * (6.2 * color + vec3(1.7)) + vec3(0.06));
    return pow(color, vec3(2.2));
}

vec3 adjustSaturation(vec3 color, float saturation) {
    float grey = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(grey), color, saturation);
}

void main() {
    vec4 albedo;
    vec3 normal;
    vec3 viewDir = normalize(ubo.cameraPos - fragPos);
    vec3 reflectDir = reflect(viewDir, normal);
    vec3 envColor;

    vec3 normalFromMap = texture(normalMap, fragTexCoord).rgb;
    normal = normalize(fragTBN * (normalFromMap * 2.0 - 1.0));

    if (pushConstants.materialType == 1) {
       albedo = texture(texSampler, fragTexCoord);
       envColor = texture(lambertianMap, normal).rgb;
       vec3 ldrColor = toneMappingFilmic(envColor); 
       ldrColor = adjustSaturation(ldrColor, 1.2);
       outColor = vec4(ldrColor * albedo.rgb, 1.0);
    } 
    else if (pushConstants.materialType == 2) 
    {
        vec3 viewDir = normalize(ubo.cameraPos - fragPos);
        vec3 reflectDir = reflect(viewDir, normal);
        envColor = texture(envMap, -reflectDir).rgb;
        vec3 ldrColor = toneMappingFilmic(envColor); 
        ldrColor = adjustSaturation(ldrColor, 1.2);
        outColor = vec4(ldrColor,0);
    } 
    else if (pushConstants.materialType == 3) 
    {
        envColor = texture(envMap, normal).rgb;  
        vec3 ldrColor = toneMappingFilmic(envColor); 
        ldrColor = adjustSaturation(ldrColor, 1.2);
        outColor = vec4(ldrColor,0);
    }

  
}

