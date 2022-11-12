// PathTrace.hlsl

#include "Helpers.hlsli"
#include "SharedDefinitions.hlsli"

RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget      : register(u0);
ByteAddressBuffer Indices[]           : register(t0, space1);
StructuredBuffer<Vertex> Vertices[]   : register(t0, space2);
StructuredBuffer<Material> Materials  : register(t0, space3);

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
ConstantBuffer<CubeConstantBuffer> g_cubeCB   : register(b1);

static const int   MaxPathLength  = 6;
static const float DistanceOnMiss = 0.0f;

// Load three 16 bit indices from a byte addressed buffer.
uint3 Load3x16BitIndices(uint offsetBytes)
{
    uint3 indices;

    // ByteAdressBuffer loads must be aligned at a 4 byte boundary.
    // Since we need to read three 16 bit indices: { 0, 1, 2 } 
    // aligned at a 4 byte boundary as: { 0 1 } { 2 0 } { 1 2 } { 0 1 } ...
    // we will load 8 bytes (~ 4 indices { a b | c d }) to handle two possible index triplet layouts,
    // based on first index's offsetBytes being aligned at the 4 byte boundary or not:
    //  Aligned:     { 0 1 | 2 - }
    //  Not aligned: { - 0 | 1 2 }
    const uint dwordAlignedOffset = offsetBytes & ~3;
    const uint2 four16BitIndices = Indices[InstanceID()].Load2(dwordAlignedOffset);

    // Aligned: { 0 1 | 2 - } => retrieve first three 16bit indices
    if (dwordAlignedOffset == offsetBytes)
    {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    }
    else // Not aligned: { - 0 | 1 2 } => retrieve last three 16bit indices
    {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }

    return indices;
}

typedef BuiltInTriangleIntersectionAttributes MyAttributes;

// Retrieve hit world position.
float3 HitWorldPosition()
{
    return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float3 HitAttribute(float3 vertexAttribute[3], BuiltInTriangleIntersectionAttributes attr)
{
    return vertexAttribute[0] +
        attr.barycentrics.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.barycentrics.y * (vertexAttribute[2] - vertexAttribute[0]);
}

// Generate a ray in world space for a camera pixel corresponding to an index from the dispatched 2D grid.
inline void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
{
    float2 xy = index + 0.5f; // center in the middle of the pixel.
    float2 screenPos = xy / DispatchRaysDimensions().xy * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates.
    screenPos.y = -screenPos.y;

    // Unproject the pixel coordinate into a ray.
    float4 world = mul(float4(screenPos, 0, 1), g_sceneCB.projectionToWorld);

    world.xyz /= world.w;
    origin = g_sceneCB.cameraPosition.xyz;
    direction = normalize(world.xyz - origin);
}

static float2 SamplePoint(in uint pixelIdx, inout uint setIdx)
{
    const uint permutation = setIdx * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx;
    return SampleCMJ2D(g_sceneCB.frameNumber, DispatchRaysDimensions().x, DispatchRaysDimensions().y, permutation);
}

// Recursive
[shader("raygeneration")]
void MyRaygenShader()
{
    float3 rayDir;
    float3 origin;

    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    GenerateCameraRay(DispatchRaysIndex().xy, origin, rayDir);

    // Trace the ray.
    // Set the ray's extents.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = rayDir;
 
    // Set TMin to a non-zero small value to avoid aliasing issues due to floating - point errors.
    // TMin should be kept small to prevent missing geometry at close contact areas.
    ray.TMin = 0.00001;
    ray.TMax = 10000.0;
    RayPayload payload = { float4(0.0f, 0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), 0, 0.0f, 0 };
    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 0, 0, ray, payload);

    // Write the raytraced color to the output texture. 
    float4 curColor = RenderTarget[DispatchRaysIndex().xy];
    float weight = (float)g_sceneCB.frameNumber / ((float)g_sceneCB.frameNumber + 1.0f);
    RenderTarget[DispatchRaysIndex().xy] = lerp(payload.color, curColor, weight);
    //RenderTarget[DispatchRaysIndex().xy] = payload.color; // debug
}

float4 TracePath(float3 hitPosition, float3 hitNormal, float3 hitTangent, int curPathLength)
{
    float2 rand01 = SamplePoint((DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x), curPathLength);
    float3 randDir = GenerateRandomRay(hitNormal, rand01.x, rand01.y);

    RayDesc newRay;
    newRay.Origin = hitPosition;
    newRay.Direction = randDir;
    newRay.TMin = 0.00001f;
    newRay.TMax = 10000.0f;

    RayPayload newPayload = { float4(0.0f, 0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), 0, 0.0f, curPathLength + 1 };

    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 0, 0, newRay, newPayload);

    float attenuation = max(0.0f, dot(randDir, hitNormal));
    return attenuation * newPayload.color;
}

[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
    float3 hitPosition = HitWorldPosition();

    // Get the base index of the triangle's first 16 bit index.
    uint indexSizeInBytes = 2;
    uint indicesPerTriangle = 3;
    uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    uint baseIndex = PrimitiveIndex() * triangleIndexStride;

    // Load up 3 16 bit indices for the triangle.
    const uint3 indices = Load3x16BitIndices(baseIndex);

    int hitIndex = InstanceID();

    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 vertexNormals[3] = {
        Vertices[hitIndex][indices[0]].normal,
        Vertices[hitIndex][indices[1]].normal,
        Vertices[hitIndex][indices[2]].normal
    };

    // Compute the triangle's normal
    float3 triangleNormal = HitAttribute(vertexNormals, attr);

    // Retrieve tangents
    float3 vertexTangents[3] = {
        Vertices[hitIndex][indices[0]].tangent,
        Vertices[hitIndex][indices[1]].tangent,
        Vertices[hitIndex][indices[2]].tangent
    };

    // Compute triangle's tangent
    float3 triangleTangent = HitAttribute(vertexTangents, attr);

    float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (payload.pathLength < MaxPathLength)
    {
        result = TracePath(hitPosition, triangleNormal, triangleTangent, payload.pathLength);
    }

    payload.color = result * Materials[InstanceIndex()].albedo + Materials[InstanceIndex()].emissive;
    payload.materialIndex = InstanceIndex();
    payload.hitNormal = triangleNormal;
    payload.tHit = RayTCurrent();
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    float4 background = float4(0.0f, 0.0f, 0.0f, 0.0f);
    payload.color = background;
    payload.hitNormal = -WorldRayDirection();
    payload.tHit = DistanceOnMiss;
}

// Iterative
float3 GenerateRandomBounce(float3 hitNormal, int curPathLength)
{
    float2 rand01 = SamplePoint((DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x), curPathLength);
    float3 randDir = GenerateRandomRay(hitNormal, rand01.x, rand01.y);
    return randDir;
}

[shader("closesthit")]
void ClosestHitShaderIterative(inout RayPayload payload, in MyAttributes attr)
{
    float3 hitPosition = HitWorldPosition();

    // Get the base index of the triangle's first 16 bit index.
    uint indexSizeInBytes = 2;
    uint indicesPerTriangle = 3;
    uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    uint baseIndex = PrimitiveIndex() * triangleIndexStride;

    // Load up 3 16 bit indices for the triangle.
    const uint3 indices = Load3x16BitIndices(baseIndex);

    int hitIndex = InstanceID();

    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 vertexNormals[3] = {
        Vertices[hitIndex][indices[0]].normal,
        Vertices[hitIndex][indices[1]].normal,
        Vertices[hitIndex][indices[2]].normal
    };

    // Compute the triangle's normal
    float3 triangleNormal = HitAttribute(vertexNormals, attr);

    // Retrieve tangents
    float3 vertexTangents[3] = {
        Vertices[hitIndex][indices[0]].tangent,
        Vertices[hitIndex][indices[1]].tangent,
        Vertices[hitIndex][indices[2]].tangent
    };

    // Compute triangle's tangent
    float3 triangleTangent = HitAttribute(vertexTangents, attr);

    //payload.color = result * Materials[InstanceIndex()].albedo + Materials[InstanceIndex()].emissive;
    payload.materialIndex = InstanceIndex();
    payload.hitNormal = triangleNormal;
    payload.tHit = RayTCurrent();
}

[shader("raygeneration")]
void RaygenShaderIterative()
{
    float3 rayDir;
    float3 origin;

    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    GenerateCameraRay(DispatchRaysIndex().xy, origin, rayDir);

    // Trace the ray.
    // Set the ray's extents.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = rayDir;

    // Set TMin to a non-zero small value to avoid aliasing issues due to floating - point errors.
    // TMin should be kept small to prevent missing geometry at close contact areas.
    ray.TMin = 0.00001;
    ray.TMax = 10000.0;
    RayPayload payload = { float4(0.0f, 0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), 0, 0.0f, 0 };

    float4 attenuation = 1.0f;
    float4 radiance = float4(0.0f, 0.0f, 0.0f, 0.0f);

    while (payload.pathLength < MaxPathLength)
    {
        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 0, 0, ray, payload);

        if (payload.tHit == DistanceOnMiss)
        {
            // Miss
            break;
        }

        // Configure new ray parameters
        ray.Origin = ray.Origin + ray.Direction * payload.tHit;
        ray.Direction = GenerateRandomBounce(payload.hitNormal, payload.pathLength);

        // Calculate and accumulate radiance contribution
        radiance += attenuation * Materials[payload.materialIndex].emissive;
        float nDotL = max(0.0, dot(ray.Direction, payload.hitNormal));
        attenuation *= nDotL * Materials[payload.materialIndex].albedo;

        ++payload.pathLength;
    }

    // Write the raytraced color to the output texture. 
    float4 curColor = RenderTarget[DispatchRaysIndex().xy];
    float weight = (float)g_sceneCB.frameNumber / ((float)g_sceneCB.frameNumber + 1.0f);
    RenderTarget[DispatchRaysIndex().xy] = lerp(radiance, curColor, weight);
    //RenderTarget[DispatchRaysIndex().xy] = radiance; // debug
}
