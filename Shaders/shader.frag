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
    vec4 cameraPos;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

layout(binding = 2) uniform sampler2D normalMap;

layout(binding = 3) uniform samplerCube envMap;

layout(binding = 4) uniform samplerCube lambertianMap;

struct Light{
    vec4 tint;
	// 0 Sun, 1 Sphere, 2 Spot
	int lightType;
	float angle;
	float strength;
	float radius;
	float power;
	float fov;
	float blend;
	float limit;
	int shadow;
};

layout(binding = 5) buffer lightBuffer {
	Light lights[];
} lightData;

layout(push_constant) uniform PushConstants {
    mat4 model;
	vec4 albedoColor;
	float roughness;
	float metalness;
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

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265359 * denom * denom;

    return nom / denom;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 gammaCorrect(vec3 color) {
    const float gamma = 2.2;
    return pow(color, vec3(1.0 / gamma));
}

float calculateSunlightEffect(float angle, vec3 lightDirection, vec3 normal) {
    float lightEffect = max(dot(normal, lightDirection), 0.0);
    lightEffect *= (angle / 3.14159265);
    return lightEffect;
}


void main() {
    vec3 albedo;
    vec3 envColor;

    if(pushConstants.albedoColor.x < 0){
       albedo = texture(texSampler, fragTexCoord).rgb; 
    } else {
       albedo = pushConstants.albedoColor.rgb;
    }

    float roughness = pushConstants.roughness;
    float metalness = pushConstants.metalness;
    
    // lambertian
    if (pushConstants.materialType == 1) {
       vec3 normalMap = texture(normalMap, fragTexCoord).rgb;
       normalMap = normalize(normalMap * 2.0 - 1.0);
       vec3 normal = normalize(fragTBN * normalMap);

       envColor = texture(lambertianMap, normal).rgb;
       vec3 ambient = toneMappingFilmic(envColor); 
       ambient = adjustSaturation(ambient, 1.2);

       vec3 diffuse = vec3(0.0);

       for(int i = 0; i < 1; i++) {
        // sun light
        if(lightData.lights[i].lightType == 0) {
           vec3 lightDir = normalize(vec3(1,1,1));
           float sunlightEffect = calculateSunlightEffect(lightData.lights[i].angle, lightDir, normal);
           float NdotL = max(dot(normal, lightDir), 0.0);
           diffuse = sunlightEffect * lightData.lights[i].tint.rgb * lightData.lights[i].strength;
           }
        }

      outColor = vec4((ambient + diffuse) * albedo, 1.0);

    } 
    // mirror
    else if (pushConstants.materialType == 2) 
    {
        vec3 normal = fragNorm;
        vec3 viewDir = normalize(ubo.cameraPos.xyz - fragPos);
        vec3 reflectDir = reflect(viewDir, normal);
        envColor = texture(envMap, -reflectDir).rgb;
        vec3 ldrColor = toneMappingFilmic(envColor); 
        ldrColor = adjustSaturation(ldrColor, 1.2);
        outColor = vec4(ldrColor,0);
    } 
    // environement
    else if (pushConstants.materialType == 3) 
    {
        vec3 normal = fragNorm;
        envColor = texture(envMap, normal).rgb;  
        vec3 ldrColor = toneMappingFilmic(envColor); 
        ldrColor = adjustSaturation(ldrColor, 1.2);
        outColor = vec4(ldrColor,0);
    } 
    // PBR
    else if (pushConstants.materialType == 4) 
    {
       vec3 normalMap = texture(normalMap, fragTexCoord).rgb;
       normalMap = normalize(normalMap * 2.0 - 1.0);
       vec3 normal = normalize(fragTBN * normalMap);

       vec3 viewDir = normalize(ubo.cameraPos.xyz - fragPos);
       vec3 lightDir = normalize(vec3(1,1,1));
       vec3 h = normalize(viewDir + lightDir);


       vec3 F0 = mix(vec3(0.04), albedo.rgb, metalness);
       vec3 F = fresnelSchlick(max(dot(h, viewDir), 0.0), F0);

       float adjustedRoughness = max(roughness, 0.05);

       float D = distributionGGX(normal, h, adjustedRoughness);    
       float G = geometrySmith(normal, viewDir, lightDir, adjustedRoughness);  

       vec3 nominator = D * F * G;
       float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
       vec3 specular = nominator / max(denominator, 0.001); 

       vec3 kS = F;
       vec3 kD = vec3(1.0) - kS;
       kD *= 1.0 - metalness;

       float NdotL = max(dot(normal, lightDir), 0.0);
       vec3 diffuse = (albedo / 3.14159265359) * kD * NdotL;

       vec3 reflectDir = reflect(viewDir, normal);
       vec3 mirrorColor = texture(envMap, -reflectDir).rgb;

       envColor = texture(lambertianMap, normal).rgb;

       vec3 mixedEnvColor = mix(envColor, mirrorColor, metalness);

       vec3 ambient = toneMappingFilmic(mixedEnvColor); 
       ambient = adjustSaturation(ambient, 1.2);

       outColor = vec4(diffuse + specular + ambient * albedo, 1.0);
    }

  
}

