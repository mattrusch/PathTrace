// SharedDefinitions.h

#ifndef SHARED_DEFINITIONS_H
#define SHARED_DEFINITIONS_H

#ifndef HLSL
typedef DirectX::XMFLOAT3 float3;
//typedef DirectX::XMFLOAT4 float4;
typedef DirectX::XMVECTOR float4;
typedef DirectX::XMMATRIX float4x4;
typedef UINT16 Index;
#endif

struct SceneConstantBuffer
{
    float4x4 projectionToWorld;
    float4 cameraPosition;
    float4 lightPosition;
    float4 lightAmbientColor;
    float4 lightDiffuseColor;
    int  frameNumber;
};

struct CubeConstantBuffer
{
    float4 albedo;
};

struct Vertex
{
    float3 position;
    float3 normal;
    float3 tangent;
};

struct Material
{
    float4 albedo;
    float4 emissive;
};

struct RayPayload
{
    float4 color;
    float3 hitNormal;
    float3 hitTangent;
    int    materialIndex;
    float  tHit;
    int    pathLength;
};

#endif//SHARED_DEFINITIONS_H
