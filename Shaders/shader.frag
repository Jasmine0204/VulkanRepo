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
    mat4 lightSpace;
    mat4 previousModel;
    vec4 cameraPos;
} ubo;


layout(binding = 1) uniform sampler2D texSampler;

layout(binding = 2) uniform sampler2D normalMap;

layout(binding = 3) uniform samplerCube envMap;

layout(binding = 4) uniform samplerCube lambertianMap;

struct Light{
    vec4 tint;
    vec4 position;
    vec4 rotation;

	// 0 Sun, 1 Sphere, 2 Spot
	int lightType;
    int shadow;

	float angle;
	float strength;
	float radius;
	float power;
	float fov;
	float blend;
	float limit;  
};

layout(binding = 5) buffer lightBuffer {
	Light lights[10];
} lightData;

layout(binding = 6) uniform sampler2D shadowMap;

layout(binding = 7) uniform sampler2D motionMap;

layout(push_constant) uniform PushConstants {
    mat4 model;
	vec4 albedoColor;
	float roughness;
	float metalness;
	int materialType;
    int lightCount;
} pushConstants;


float calculateShadow(vec4 fragPosLightSpace){
    // Citation: shadow related part of code is inspired by https://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/#setting-up-the-rendertarget-and-the-mvp-matrix
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // PCF
    float shadow = 0.0;
    float texelSize = 1.0 / textureSize(shadowMap, 0).x;
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += projCoords.z > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    return shadow;
}

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

vec3 computeRepresentativePoint(vec3 lightPos, float lightRadius, vec3 fragPos, vec3 viewDir, vec3 normal) {
    vec3 O = fragPos; // origin
    vec3 D = reflect(-viewDir, normal); // direction
    vec3 C = lightPos; // sphere origin
    float r = lightRadius; // sphere radius

    // Citation: Refer to the basic idea of ray-sphere intersection in this tutorial https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection
    vec3 L = O - C;
    float a = dot(D, D);
    float b = 2.0 * dot(D, L);
    float c = dot(L, L) - r * r;

    float discriminant = b*b - 4.0*a*c;

    if (discriminant < 0.0) {
        return vec3(0.0); // no root
    }

    float t1 = (-b - sqrt(discriminant)) / (2.0 * a);
    float t2 = (-b + sqrt(discriminant)) / (2.0 * a);

    float t = min(t1, t2);
    if (t < 0.0) t = max(t1, t2); 
    if (t < 0.0) {
        return vec3(0.0);
    }

    vec3 P = O + t * D;
    return P;
}


void main() {
    vec3 albedo;
    vec3 envColor;
    vec4 fragPosLightSpace = ubo.lightSpace * vec4(fragPos, 1.0);
    float shadow = calculateShadow(fragPosLightSpace);

    if(pushConstants.albedoColor.x < 0){
       albedo = texture(texSampler, fragTexCoord).rgb; 
    } else {
       albedo = pushConstants.albedoColor.rgb;
    }

    float roughness = pushConstants.roughness;
    float metalness = pushConstants.metalness;

    // diffuse mat
    if (pushConstants.materialType == 1) {
       vec3 normalMap = texture(normalMap, fragTexCoord).rgb;
       normalMap = normalize(normalMap * 2.0 - 1.0);
       vec3 normal = normalize(fragTBN * normalMap);

       envColor = texture(lambertianMap, normal).rgb;
       vec3 ambient = toneMappingFilmic(envColor); 
       ambient = adjustSaturation(ambient, 1.2);

       vec3 diffuse = vec3(0.0);

       for(int i = 0; i < pushConstants.lightCount; i++) {

        Light light = lightData.lights[i];

        vec3 lightDir;
        float distance;
        float attenuation;

        // sun light
        if(light.lightType == 0) {
           lightDir = normalize(vec3(1,1,1));
           float NdotL = max(dot(normal, lightDir), 0.0);
           diffuse = diffuse + NdotL * light.tint.rgb * light.strength * (1.0 + light.angle / 3.14159265);
           } 
        // sphere light
        else if (light.lightType == 1){
          lightDir = normalize(light.position.xyz - fragPos);
           distance = length(light.position.xyz - fragPos);
           float radius = light.radius;

           float effectiveDistance = max(distance - radius, 0.0);
           attenuation = max(0.0, 1.0 - pow((effectiveDistance / light.limit), 4));
            
           float NdotL = max(dot(normal, lightDir), 0.0);
           diffuse += NdotL * light.tint.rgb * light.power * attenuation;
           }
        // spot light
        else if (light.lightType == 2){
            vec3 lightDir = normalize(light.position.xyz - fragPos);
            float distance = length(light.position.xyz - fragPos);
            vec3 spotLightDir = normalize(light.rotation.xyz);
            float theta = dot(lightDir, normalize(-spotLightDir));

            float epsilon = cos(light.fov) * (1.0 - light.blend) + light.blend;
            float intensity = smoothstep(epsilon, 1.0, theta);

            float attenuation = max(0.0, 1.0 - pow((distance - light.radius) / light.limit, 4.0));
            float NdotL = max(dot(normal, lightDir), 0.0);
            diffuse += NdotL * light.tint.rgb * light.power * attenuation * intensity;
        }
       }
      diffuse *= (1.0f - shadow);
      outColor = vec4((ambient + diffuse) * albedo, 1.0);

    } 
    // mirror
    else if (pushConstants.materialType == 2) 
    {
        vec3 normal = fragNorm;
        vec3 viewDir = normalize(fragPos - ubo.cameraPos.xyz);
        vec3 reflectDir = reflect(viewDir, normal);
        envColor = texture(envMap, reflectDir).rgb;
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
       vec3 lightDir;

       vec3 diffuse = vec3(0.0);
       vec3 specular = vec3(0.0);
       vec3 kD;

       for(int i = 0; i < pushConstants.lightCount; i++) {
        Light light = lightData.lights[i];

        float distance;
        float attenuation;

        // Citation: The code structure of different light types is inspired by LearnOpenGL https://learnopengl.com/Lighting/Light-casters
        // sun light
        if(light.lightType == 0) {
           lightDir = normalize(vec3(1,1,1));
           float NdotL = max(dot(normal, lightDir), 0.0);
           diffuse += NdotL * light.tint.rgb * light.strength * (1.0 + light.angle / 3.14159265);

           vec3 h = normalize(viewDir + lightDir);

           vec3 F0 = mix(vec3(0.04), albedo.rgb, metalness);
           vec3 F = fresnelSchlick(max(dot(h, viewDir), 0.0), F0);

           float adjustedRoughness = max(roughness, 0.05);

           float D = distributionGGX(normal, h, adjustedRoughness);    
           float G = geometrySmith(normal, viewDir, lightDir, adjustedRoughness);  

           vec3 nominator = D * F * G;
           float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
           specular += nominator / max(denominator, 0.001); 

           vec3 kS = F;
           kD = vec3(1.0) - kS;
           kD *= 1.0 - metalness;
        } 
        // sphere light
        else if (light.lightType == 1){
           lightDir = normalize(light.position.xyz - fragPos);
           distance = length(light.position.xyz - fragPos);
           float radius = light.radius;

           float effectiveDistance = max(distance - radius, 0.0);
           attenuation = max(0.0, 1.0 - pow((effectiveDistance / light.limit), 4));
            
           float NdotL = max(dot(normal, lightDir), 0.0);
           diffuse += NdotL * light.tint.rgb * light.power * attenuation;

           // compute new light dir for specular
           vec3 representativePoint = normalize(computeRepresentativePoint(light.position.xyz, light.radius, fragPos, viewDir, normal));
           lightDir = normalize(representativePoint - fragPos);

           vec3 h = normalize(viewDir + lightDir);

           vec3 F0 = mix(vec3(0.04), albedo.rgb, metalness);
           vec3 F = fresnelSchlick(max(dot(h, viewDir), 0.0), F0);

           float adjustedRoughness = max(roughness, 0.05);

           float D = distributionGGX(normal, h, adjustedRoughness);    
           float G = geometrySmith(normal, viewDir, lightDir, adjustedRoughness);  

           vec3 nominator = D * F * G;
           float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
           specular += nominator / max(denominator, 0.001) * light.tint.rgb * light.power * attenuation; 

           vec3 kS = F;
           kD = vec3(1.0) - kS;
           kD *= 1.0 - metalness;
    
           }
        // spot light
        else if (light.lightType == 2){
            vec3 lightDir = normalize(light.position.xyz - fragPos);
            float distance = length(light.position.xyz - fragPos);
            vec3 spotLightDir = normalize(light.rotation.xyz);
            float theta = dot(lightDir, -spotLightDir);

            float epsilon = cos(light.fov) * (1.0 - light.blend) + light.blend;
            float intensity = smoothstep(epsilon, 1.0, theta);

            float attenuation = max(0.0, 1.0 - pow((distance - light.radius) / light.limit, 4.0));
            float NdotL = max(dot(normal, lightDir), 0.0);
            diffuse += NdotL * light.tint.rgb * light.power * attenuation * intensity;

            vec3 h = normalize(viewDir + lightDir);

           vec3 F0 = mix(vec3(0.04), albedo.rgb, metalness);
           vec3 F = fresnelSchlick(max(dot(h, viewDir), 0.0), F0);

           float adjustedRoughness = max(roughness, 0.05);

           float D = distributionGGX(normal, h, adjustedRoughness);    
           float G = geometrySmith(normal, viewDir, lightDir, adjustedRoughness);  

           vec3 nominator = D * F * G;
           float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
           specular += nominator / max(denominator, 0.001) * light.tint.rgb * light.power * attenuation * intensity; 

           vec3 kS = F;
           kD = vec3(1.0) - kS;
           kD *= 1.0 - metalness;
        }
     }

       diffuse *= (albedo / 3.14159265) * kD;
       vec3 reflectDir = reflect(viewDir, normal);
       vec3 mirrorColor = texture(envMap, -reflectDir).rgb;

       envColor = texture(lambertianMap, normal).rgb;

       vec3 mixedEnvColor = mix(envColor, mirrorColor, metalness);

       vec3 ambient = toneMappingFilmic(mixedEnvColor); 
       ambient = adjustSaturation(ambient, 1.2);

       
       //diffuse *= (1.0f - shadow);
       //specular *= (1.0f - shadow);

       outColor = vec4(diffuse + specular + ambient * albedo, 1.0);
    }
}

