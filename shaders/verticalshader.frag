#version 450

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;


layout(binding = 7) uniform sampler2D motionMap;

layout(binding = 9) uniform sampler2D horizontalImage;


void main() {

    vec4 blurColor = texture(horizontalImage, fragTexCoord);

    // motion blur texture
    vec2 motionVector = texture(motionMap, fragTexCoord).xy * 2.0 - 1.0;
    
      // motion blur
      float totalWeight = 1.0;
      int samples = 5;
      float blurScale = 0.03;

      for(int i = 1; i <= samples; ++i) {
        float sampleWeight = 1.0 - (float(i) / samples);
        float offset = motionVector.y * blurScale * float(i);

        blurColor += texture(horizontalImage, fragTexCoord + vec2(offset, 0.0)) * sampleWeight;
        blurColor += texture(horizontalImage, fragTexCoord - vec2(offset, 0.0)) * sampleWeight;
        totalWeight += 2.0 * sampleWeight;
    }

    blurColor /= totalWeight;
    outColor = blurColor;

    //outColor = vec4(motionVector, 0.0, 1.0);

}

