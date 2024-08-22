'''
// #define ANIMATE_NOISE 1

const float c_pi = 3.14159265359f;
const float c_goldenRatioConjugate = 0.61803398875f; // also just fract(goldenRatio)

const float c_FOV = 90.0f; // in degrees
const float c_cameraDistance = 80.0f;
const float c_minCameraAngle = 0.0f;
const float c_maxCameraAngle = c_pi;
const vec3 c_cameraAt = vec3(0.0f, 40.0f, 0.0f);
const float c_rayMaxDist = 10000.0f;

#define c_lightDir normalize(vec3(0.0f, 0.0f, 0.2f))
const vec3 c_lightColor = vec3(1.0f, 0.8f, 0.5f);
const vec3 c_lightAmbient = vec3(0.05f, 0.05f, 0.05f);

const vec2 c_defaultMousePos = vec2(200.0f / 800.0f, 275.0f / 450.0f);

const float c_hitNormalNudge = 0.1f;

const int c_numRayMarchSteps = 16;

const float c_fogDensity = 0.002f;
const vec3 c_fogColorLit = vec3(1.0f, 1.0f, 1.0f);
const vec3 c_fogColorUnlit = vec3(0.0f, 0.0f, 0.0f);

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
uniform float     iTimeDelta;            // render time (in seconds)
uniform float     iFrameRate;            // shader frame rate
uniform int       iFrame;                // shader playback frame
uniform float     iChannelTime[4];       // channel playback time (in seconds)
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform samplerXX iChannel0..3;          // input channel. XX = 2D/Cube
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)

// ACES tone mapping curve fit to go from HDR to LDR
//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0f, 1.0f);
}

vec3 LessThan(vec3 f, float value)
{
    return vec3(
        (f.x < value) ? 1.0f : 0.0f,
        (f.y < value) ? 1.0f : 0.0f,
        (f.z < value) ? 1.0f : 0.0f);
}

vec3 LinearToSRGB(vec3 rgb)
{
    rgb = clamp(rgb, 0.0f, 1.0f);
    
    return mix(
        pow(rgb * 1.055f, vec3(1.f / 2.4f)) - 0.055f,
        rgb * 12.92f,
        LessThan(rgb, 0.0031308f)
    );
}

// this noise, including the 5.58... scrolling constant are from Jorge Jimenez
float InterleavedGradientNoise(vec2 pixel, int frame) 
{
    pixel += (float(frame) * 5.588238f);
    return fract(52.9829189f * fract(0.06711056f*float(pixel.x) + 0.00583715f*float(pixel.y)));  
}

// from https://www.shadertoy.com/view/4djSRW
float hash13(vec3 p3)
{
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void GetCameraVectors(out vec3 cameraPos, out vec3 cameraFwd, out vec3 cameraUp, out vec3 cameraRight)
{   
    vec2 mouse = iMouse.xy;
    if (dot(mouse, vec2(1.0f, 1.0f)) == 0.0f)
        mouse = c_defaultMousePos * iResolution.xy;    
    
    float angleX = -mouse.x * 16.0f / float(iResolution.x);
    float angleY = mix(c_minCameraAngle, c_maxCameraAngle, mouse.y / float(iResolution.y));
    
    cameraPos.x = sin(angleX) * sin(angleY) * c_cameraDistance;
    cameraPos.y = -cos(angleY) * c_cameraDistance;
    cameraPos.z = cos(angleX) * sin(angleY) * c_cameraDistance;
    
    cameraPos += c_cameraAt;
    
    cameraFwd = normalize(c_cameraAt - cameraPos);
    cameraRight = normalize(cross(cameraFwd, vec3(0.0f, 1.0f, 0.0f)));
    cameraUp = normalize(cross(cameraRight, cameraFwd));   
}

struct SRayHitInfo
{
    float dist;
    vec3 normal;
    vec3 diffuse;
};
    
bool RayVsSphere(in vec3 rayPos, in vec3 rayDir, inout SRayHitInfo info, in vec4 sphere, in vec3 diffuse)
{
	//get the vector from the center of this sphere to where the ray begins.
	vec3 m = rayPos - sphere.xyz;

    //get the dot product of the above vector and the ray`s vector
	float b = dot(m, rayDir);

	float c = dot(m, m) - sphere.w * sphere.w;

	//exit if r`s origin outside s (c > 0) and r pointing away from s (b > 0)
	if(c > 0.0 && b > 0.0)
		return false;

	//calculate discriminant
	float discr = b * b - c;

	//a negative discriminant corresponds to ray missing sphere
	if(discr < 0.0)
		return false;
    
	//ray now found to intersect sphere, compute smallest t value of intersection
    bool fromInside = false;
	float dist = -b - sqrt(discr);
    if (dist < 0.0f)
    {
        fromInside = true;
        dist = -b + sqrt(discr);
    }
    
	if (dist > 0.0f && dist < info.dist)
    {
        info.dist = dist;        
        info.normal = normalize((rayPos+rayDir*dist) - sphere.xyz) * (fromInside ? -1.0f : 1.0f);
		info.diffuse = diffuse;        
        return true;
    }
    
    return false;
}
    
bool RayVsPlane(in vec3 rayPos, in vec3 rayDir, inout SRayHitInfo info, in vec4 plane, in vec3 diffuse)
{
    float dist = -1.0f;
    float denom = dot(plane.xyz, rayDir);
    if (abs(denom) > 0.001f)
    {
        dist = (plane.w - dot(plane.xyz, rayPos)) / denom;
    
        if (dist > 0.0f && dist < info.dist)
        {
            info.dist = dist;        
            info.normal = plane.xyz;
            info.diffuse = diffuse;
            return true;
        }
    }
    return false;
}

float ScalarTriple(vec3 u, vec3 v, vec3 w)
{
    return dot(cross(u, v), w);
}

bool RayVsQuad(in vec3 rayPos, in vec3 rayDir, inout SRayHitInfo info, in vec3 a, in vec3 b, in vec3 c, in vec3 d, in vec3 diffuse, bool doubleSided)
{
    // calculate normal and flip vertices order if needed
    vec3 normal = normalize(cross(c-a, c-b));
    if (doubleSided && dot(normal, rayDir) > 0.0f)
    {
        normal *= -1.0f;
        
		vec3 temp = d;
        d = a;
        a = temp;
        
        temp = b;
        b = c;
        c = temp;
    }
    
    vec3 p = rayPos;
    vec3 q = rayPos + rayDir;
    vec3 pq = q - p;
    vec3 pa = a - p;
    vec3 pb = b - p;
    vec3 pc = c - p;
    
    // determine which triangle to test against by testing against diagonal first
    vec3 m = cross(pc, pq);
    float v = dot(pa, m);
    vec3 intersectPos;
    if (v >= 0.0f)
    {
        // test against triangle a,b,c
        float u = -dot(pb, m);
        if (u < 0.0f) return false;
        float w = ScalarTriple(pq, pb, pa);
        if (w < 0.0f) return false;
        float denom = 1.0f / (u+v+w);
        u*=denom;
        v*=denom;
        w*=denom;
        intersectPos = u*a+v*b+w*c;
    }
    else
    {
        vec3 pd = d - p;
        float u = dot(pd, m);
        if (u < 0.0f) return false;
        float w = ScalarTriple(pq, pa, pd);
        if (w < 0.0f) return false;
        v = -v;
        float denom = 1.0f / (u+v+w);
        u*=denom;
        v*=denom;
        w*=denom;
        intersectPos = u*a+v*d+w*c;
    }
    
    float dist;
    if (abs(rayDir.x) > 0.1f)
    {
        dist = (intersectPos.x - rayPos.x) / rayDir.x;
    }
    else if (abs(rayDir.y) > 0.1f)
    {
        dist = (intersectPos.y - rayPos.y) / rayDir.y;
    }
    else
    {
        dist = (intersectPos.z - rayPos.z) / rayDir.z;
    }
    
	if (dist > 0.0f && dist < info.dist)
    {
        info.dist = dist;        
        info.normal = normal;
		info.diffuse = diffuse;        
        return true;
    }    
    
    return false;
}

SRayHitInfo RayVsScene(in vec3 rayPos, in vec3 rayDir)
{
    SRayHitInfo hitInfo;
    hitInfo.dist = c_rayMaxDist;

    // the floor
    if(RayVsPlane(rayPos, rayDir, hitInfo, vec4(0.0f, 1.0f, 0.0f, 0.0f), vec3(0.2f, 0.2f, 0.2f)))
    {
        // uncomment this for a checkerboard floor
        /*
        vec3 hitPos = rayPos + rayDir * hitInfo.dist;
        vec2 uv = floor(hitPos.xz / 100.0f);
        float shade = mix(0.6f, 0.2f, mod(uv.x + uv.y, 2.0f));
        hitInfo.diffuse = vec3(shade, shade, shade);
		*/
    }
    
    // some floating spheres to cast shadows
    RayVsSphere(rayPos, rayDir, hitInfo, vec4(-60.0f, 40.0f, 0.0f, 10.0f), vec3(1.0f, 0.0f, 1.0f));
    RayVsSphere(rayPos, rayDir, hitInfo, vec4(-30.0f, 40.0f, 0.0f, 10.0f), vec3(1.0f, 0.0f, 0.0f));
    RayVsSphere(rayPos, rayDir, hitInfo, vec4(0.0f, 40.0f, 0.0f, 10.0f), vec3(0.0f, 1.0f, 0.0f));
    RayVsSphere(rayPos, rayDir, hitInfo, vec4(30.0f, 40.0f, 0.0f, 10.0f), vec3(0.0f, 0.0f, 1.0f));
    RayVsSphere(rayPos, rayDir, hitInfo, vec4(60.0f, 40.0f, 0.0f, 10.0f), vec3(1.0f, 1.0f, 0.0f));
    
    // back wall
    {
        vec3 scale = vec3(100.0f, 40.0f, 1.0f);
        vec3 offset = vec3(0.0f, 0.0f, 10.0f);
        vec3 A = vec3(-1.0f, 0.0f, 0.0f) * scale + offset;
        vec3 B = vec3(-1.0f, 1.0f, 0.0f) * scale + offset;
        vec3 C = vec3( 1.0f, 1.0f, 0.0f) * scale + offset;
        vec3 D = vec3( 1.0f, 0.0f, 0.0f) * scale + offset;
    	RayVsQuad(rayPos, rayDir, hitInfo, A, B, C, D, vec3(1.0f, 0.0f, 1.0f), true);
	}     
    
    return hitInfo;
}


vec3 GetColorForRay(in vec3 rayPos, in vec3 rayDir, out float hitDistance)
{
    // trace primary ray
	SRayHitInfo hitInfo = RayVsScene(rayPos, rayDir);
    
    // set the hitDistance out parameter
    hitDistance = hitInfo.dist;
    
    if (hitInfo.dist == c_rayMaxDist)
        return texture(iChannel0, rayDir).rgb;
    
    // trace shadow ray
    vec3 hitPos = rayPos + rayDir * hitInfo.dist;
    hitPos += hitInfo.normal * c_hitNormalNudge;
    SRayHitInfo shadowHitInfo = RayVsScene(hitPos, c_lightDir);
    float shadowTerm = (shadowHitInfo.dist == c_rayMaxDist) ? 1.0f : 0.0f;
    
    // do diffuse lighting
    float dp = clamp(dot(hitInfo.normal, c_lightDir), 0.0f, 1.0f);
	return c_lightAmbient * hitInfo.diffuse + dp * hitInfo.diffuse * c_lightColor * shadowTerm;
}

// ray march from the camera to the depth of what the ray hit to do some simple scattering
vec3 ApplyFog(in vec3 rayPos, in vec3 rayDir, in vec3 pixelColor, in float rayHitTime, in int panel, in vec2 pixelPos)
{         
    // Offset the start of the ray between 0 and 1 ray marching steps.
    // This turns banding into noise.
    int frame = 0;
    #if ANIMATE_NOISE
    	frame = iFrame % 64;
    #endif
    
    float startRayOffset = 0.0f;
    if (panel == 0)
    {
        startRayOffset = 0.5f;
    }
    else if (panel == 1)
    {
        // white noise
        startRayOffset = hash13(vec3(pixelPos, float(frame)));
    }
    else if (panel == 2)
    {
        // blue noise
        startRayOffset = texture(iChannel1, pixelPos / 1024.0f).r;
        startRayOffset = fract(startRayOffset + float(frame) * c_goldenRatioConjugate);
    }    
    else if (panel == 3)
    {
        // interleaved gradient noise
        startRayOffset = InterleavedGradientNoise(pixelPos, frame);
    }
    
    // calculate how much of the ray is in direct light by taking a fixed number of steps down the ray
    // and calculating the percent.
    // Note: in a rasterizer, you`d replace the RayVsScene raytracing with a shadow map lookup!
    float fogLitPercent = 0.0f;
    for (int i = 0; i < c_numRayMarchSteps; ++i)
    {
        vec3 testPos = rayPos + rayDir * rayHitTime * ((float(i)+startRayOffset) / float(c_numRayMarchSteps));
        SRayHitInfo shadowHitInfo = RayVsScene(testPos, c_lightDir);
        fogLitPercent = mix(fogLitPercent, (shadowHitInfo.dist == c_rayMaxDist) ? 1.0f : 0.0f, 1.0f / float(i+1));
    }
    
    vec3 fogColor = mix(c_fogColorUnlit, c_fogColorLit, fogLitPercent);
    float absorb = exp(-rayHitTime * c_fogDensity);
    return mix(fogColor, pixelColor, absorb);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // get the camera vectors
    vec3 cameraPos, cameraFwd, cameraUp, cameraRight;
    GetCameraVectors(cameraPos, cameraFwd, cameraUp, cameraRight);    
    
    // calculate the ray direction for this pixel
    vec2 uv = fragCoord/iResolution.xy;
	float aspectRatio = iResolution.x / iResolution.y;
    int panel = 0;
    vec3 rayDir;
    {   
        panel = int(dot(floor(uv*2.0f), vec2(1.0f, 2.0f)));
        
		vec2 screen = fract(uv*2.0f) * 2.0f - 1.0f;
        screen.y /= aspectRatio;
                
        float cameraDistance = tan(c_FOV * 0.5f * c_pi / 180.0f);       
        rayDir = vec3(screen, cameraDistance);
        rayDir = normalize(mat3(cameraRight, cameraUp, cameraFwd) * rayDir);
    }
    
    // do rendering for this pixel
    float rayHitTime;
    vec3 pixelColor = GetColorForRay(cameraPos, rayDir, rayHitTime);
    
    // apply fog
    pixelColor = ApplyFog(cameraPos, rayDir, pixelColor, rayHitTime, panel, fragCoord);
    
    // tone map the color to bring it from unbound HDR levels to SDR levels
    pixelColor = ACESFilm(pixelColor);
    
    // convert to sRGB, then output
    pixelColor = LinearToSRGB(pixelColor);
    fragColor = vec4(pixelColor, 1.0f);        
}
'''