// PathTrace.cpp

#include "stdafx.h"
#include "PathTrace.h"
#include "DeviceResources.h"
#include "DirectXRaytracingHelper.h"
#include "CompiledShaders\PathTrace.hlsl.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "tiny_gltf.h"

using namespace DirectX;

const wchar_t* PathTrace::c_hitGroupName = L"MyHitGroup";
#if 0
const wchar_t* PathTrace::c_raygenShaderName = L"MyRaygenShader";
const wchar_t* PathTrace::c_closestHitShaderName = L"MyClosestHitShader";
#else
const wchar_t* PathTrace::c_raygenShaderName = L"RaygenShaderIterative";
const wchar_t* PathTrace::c_closestHitShaderName = L"ClosestHitShaderIterative";
#endif
const wchar_t* PathTrace::c_missShaderName = L"MyMissShader";

PathTrace::PathTrace(UINT width, UINT height, std::wstring name)
	: DXSample(width, height, name)
    , m_raytracingOutputResourceUAVDescriptorHeapIndex(UINT_MAX)
    , m_curRotationAngleRad(0.0f)
    , m_eye()
    , m_at()
    , m_up()
{
	UpdateForSizeChange(width, height);
}

void PathTrace::OnDeviceLost()
{
    ReleaseWindowSizeDependentResources();
    ReleaseDeviceDependentResources();
}

void PathTrace::OnDeviceRestored()
{
    CreateDeviceDependentResources();
    CreateWindowSizeDependentResources();
}

void PathTrace::OnInit()
{
    m_deviceResources = std::make_unique<DX::DeviceResources>(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_UNKNOWN,
        FrameCount,
        D3D_FEATURE_LEVEL_11_0,
        // Sample shows handling of use cases with tearing support, which is OS dependent and has been supported since TH2.
        // Since the sample requires build 1809 (RS5) or higher, we don't need to handle non-tearing cases.
        DX::DeviceResources::c_RequireTearingSupport,
        m_adapterIDoverride
        );
    m_deviceResources->RegisterDeviceNotify(this);
    m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
    m_deviceResources->InitializeDXGIAdapter();

    ThrowIfFalse(IsDirectXRaytracingSupported(m_deviceResources->GetAdapter()),
        L"ERROR: DirectX Raytracing is not supported by your OS, GPU and/or driver.\n\n");

    m_deviceResources->CreateDeviceResources();
    m_deviceResources->CreateWindowSizeDependentResources();

    InitializeScene();

    CreateDeviceDependentResources();
    CreateWindowSizeDependentResources();
}

// Update camera matrices passed into the shader.
void PathTrace::UpdateCameraMatrices()
{
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    m_sceneCB[frameIndex].cameraPosition = m_eye;
    float fovAngleY = 45.0f;
    XMMATRIX view = XMMatrixLookAtLH(m_eye, m_at, m_up);
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(fovAngleY), m_aspectRatio, 1.0f, 125.0f);
    XMMATRIX viewProj = view * proj;

    m_sceneCB[frameIndex].projectionToWorld = XMMatrixInverse(nullptr, viewProj);

    m_sceneCB[frameIndex].frameNumber = m_frameNumber;
}

void PathTrace::UpdateRotation(float deltaX, float deltaY)
{
    XMVECTOR right = XMVector3Cross((m_eye - m_at), m_up);
    XMMATRIX rotate = XMMatrixRotationY(deltaY) * XMMatrixRotationAxis(right, deltaX);
    m_eye = XMVector3Transform(m_eye, rotate);
    m_up = XMVector3Transform(m_up, rotate);
    m_at = XMVector3Transform(m_at, rotate);
    m_frameNumber = 0;
}

void PathTrace::UpdatePosition(float forward)
{
    XMVECTOR forwardVec = XMVector3Normalize(m_at - m_eye);
    XMVECTOR rightVec = XMVector3Cross(forwardVec, m_up);
    XMVECTOR yAxis = { 0.0f, 1.0f, 0.0f };
    XMVECTOR xzPlaneVec = forward * XMVector3Cross(yAxis, rightVec);

    m_eye += xzPlaneVec;
    m_up += xzPlaneVec;
    m_at += xzPlaneVec;
    m_frameNumber = 0;
}

void PathTrace::InitializeScene()
{
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    // Setup camera.
    {
        // Initialize the view and projection inverse matrices.
        m_eye = { 0.0f, 4.0f, -10.0f, 1.0f };
        m_at = { 0.0f, 0.0f, 0.0f, 1.0f };
        XMVECTOR right = { 1.0f, 0.0f, 0.0f, 0.0f };

        XMVECTOR direction = XMVector4Normalize(m_at - m_eye);
        m_up = XMVector3Normalize(XMVector3Cross(direction, right));

        // Rotate camera around Y axis.
        XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(0.0f));
        m_eye = XMVector3Transform(m_eye, rotate);
        m_up = XMVector3Transform(m_up, rotate);

        UpdateCameraMatrices();
    }

    // Apply the initial values to all frames' buffer instances.
    for (auto& sceneCB : m_sceneCB)
    {
        sceneCB = m_sceneCB[frameIndex];
    }
}

static float GetElapsedTime()
{
    static ULONGLONG lastTime = GetTickCount64();
    ULONGLONG curTime = GetTickCount64();
    float elapsedTime = static_cast<float>(curTime - lastTime) * 0.001f;
    lastTime = curTime;

    return elapsedTime;
}

void PathTrace::OnUpdate()
{
    float elapsedTime = GetElapsedTime();

    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
    auto prevFrameIndex = m_deviceResources->GetPreviousFrameIndex();

    // Rotate the camera around Y axis.
    {
        XMMATRIX rotate = XMMatrixIdentity();
        m_eye = XMVector3Transform(m_eye, rotate);
        m_up = XMVector3Transform(m_up, rotate);
        m_at = XMVector3Transform(m_at, rotate);

        UpdateCameraMatrices();

        m_frameNumber++;
    }
}

void PathTrace::OnRender()
{
    if (!m_deviceResources->IsWindowVisible())
    {
        return;
    }

    m_deviceResources->Prepare();

    DoRaytracing();
    CopyRaytracingOutputToBackbuffer();

    m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT);
}

void PathTrace::OnSizeChanged(UINT width, UINT height, bool minimized)
{
    if (!m_deviceResources->WindowSizeChanged(width, height, minimized))
    {
        return;
    }

    UpdateForSizeChange(width, height);
    ReleaseWindowSizeDependentResources();
    CreateWindowSizeDependentResources();
}

void PathTrace::OnDestroy()
{
    m_deviceResources->WaitForGpu();
    OnDeviceLost();
}

// Create resources that are dependent on the size of the main window.
void PathTrace::CreateWindowSizeDependentResources()
{
    CreateRaytracingOutputResource();
    UpdateCameraMatrices();
}

void PathTrace::ReleaseWindowSizeDependentResources()
{
    m_raytracingOutput.Reset();
}

void PathTrace::CreateDeviceDependentResources()
{
    // Initialize raytracing pipeline.

    // Create raytracing interfaces: raytracing device and commandlist.
    CreateRaytracingInterfaces();

    // Create root signatures for the shaders.
    CreateRootSignatures();

    // Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
    CreateRaytracingPipelineStateObject();

    // Create a heap for descriptors.
    CreateDescriptorHeap();

    // Build geometry to be used in the sample.
    BuildGeometry();

    // Build raytracing acceleration structures from the generated geometry.
    BuildAccelerationStructures();

    // Create constant buffers for the geometry and the scene.
    CreateConstantBuffers();

    // Build shader tables, which define shaders and their local root arguments.
    BuildShaderTables();

    // Create an output 2D texture to store the raytracing result to.
    CreateRaytracingOutputResource();
}

void PathTrace::ReleaseDeviceDependentResources()
{
    m_raytracingGlobalRootSignature.Reset();
    m_raytracingLocalRootSignature.Reset();

    m_dxrDevice.Reset();
    m_dxrCommandList.Reset();
    m_dxrStateObject.Reset();

    m_descriptorHeap.Reset();
    m_descriptorsAllocated = 0;
    m_raytracingOutputResourceUAVDescriptorHeapIndex = UINT_MAX;

    for (auto& indexBuffer : m_indexBuffers)
    {
        indexBuffer.resource.Reset();
    }

    for (auto& vertexBuffer : m_vertexBuffers)
    {
        vertexBuffer.resource.Reset();
    }

    m_materialBuffer.resource.Reset();
    m_perFrameConstants.Reset();
    m_rayGenShaderTable.Reset();
    m_missShaderTable.Reset();
    m_hitGroupShaderTable.Reset();

    for (auto& blas : m_bottomLevelAccelerationStructures)
    {
        blas.Reset();
    }

    m_topLevelAccelerationStructure.Reset();
}

void PathTrace::CreateRaytracingInterfaces()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

    ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_dxrDevice)), L"Couldn't get DirectX Raytracing interface for the device.\n");
    ThrowIfFailed(commandList->QueryInterface(IID_PPV_ARGS(&m_dxrCommandList)), L"Couldn't get DirectX Raytracing interface for the command list.\n");
}

void PathTrace::SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig)
{
    auto device = m_deviceResources->GetD3DDevice();
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;

    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), error ? static_cast<wchar_t*>(error->GetBufferPointer()) : nullptr);
    ThrowIfFailed(device->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&(*rootSig))));
}

void PathTrace::CreateRootSignatures()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Global Root Signature
    // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
    {
        CD3DX12_DESCRIPTOR_RANGE ranges[4]; // Perfomance TIP: Order from most frequent to least frequent.
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);     // output texture
        ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 1);  // static index buffers
        ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 2);  // static vertex buffers
        ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 3);  // static vertex buffers

        CD3DX12_ROOT_PARAMETER rootParameters[GlobalRootSignatureParams::Count];
        rootParameters[GlobalRootSignatureParams::OutputViewSlot].InitAsDescriptorTable(1, &ranges[0]);
        rootParameters[GlobalRootSignatureParams::AccelerationStructureSlot].InitAsShaderResourceView(0);
        rootParameters[GlobalRootSignatureParams::SceneConstantSlot].InitAsConstantBufferView(0);
        rootParameters[GlobalRootSignatureParams::VertexBuffersSlot].InitAsDescriptorTable(3, &ranges[1]);
        CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
        SerializeAndCreateRaytracingRootSignature(globalRootSignatureDesc, &m_raytracingGlobalRootSignature);
    }

    // Local Root Signature
    // This is a root signature that enables a shader to have unique arguments that come from shader tables.
    {
        CD3DX12_ROOT_PARAMETER rootParameters[LocalRootSignatureParams::Count];
        rootParameters[LocalRootSignatureParams::CubeConstantSlot].InitAsConstants(SizeOfInUint32(m_cubeCB), 1);
        CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
        localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
        SerializeAndCreateRaytracingRootSignature(localRootSignatureDesc, &m_raytracingLocalRootSignature);
    }
}

void PathTrace::CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    // Ray gen and miss shaders in this sample are not using a local root signature and thus one is not associated with them.

    // Local root signature to be used in a hit group.
    auto localRootSignature = raytracingPipeline->CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
    localRootSignature->SetRootSignature(m_raytracingLocalRootSignature.Get());
    // Define explicit shader association for the local root signature. 
    {
        auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
        rootSignatureAssociation->AddExport(c_hitGroupName);
    }
}

void PathTrace::CreateRaytracingPipelineStateObject()
{
    // Create 7 subobjects that combine into a RTPSO:
    // Subobjects need to be associated with DXIL exports (i.e. shaders) either by way of default or explicit associations.
    // Default association applies to every exported shader entrypoint that doesn't have any of the same type of subobject associated with it.
    // This simple sample utilizes default shader association except for local root signature subobject
    // which has an explicit association specified purely for demonstration purposes.
    // 1 - DXIL library
    // 1 - Triangle hit group
    // 1 - Shader config
    // 2 - Local root signature and association
    // 1 - Global root signature
    // 1 - Pipeline config
    CD3DX12_STATE_OBJECT_DESC raytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };


    // DXIL library
    // This contains the shaders and their entrypoints for the state object.
    // Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
    auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void*)g_pPathTrace, ARRAYSIZE(g_pPathTrace));
    lib->SetDXILLibrary(&libdxil);
    // Define which shader exports to surface from the library.
    // If no shader exports are defined for a DXIL library subobject, all shaders will be surfaced.
    // In this sample, this could be ommited for convenience since the sample uses all shaders in the library. 
    {
        lib->DefineExport(c_raygenShaderName);
        lib->DefineExport(c_closestHitShaderName);
        lib->DefineExport(c_missShaderName);
    }

    // Triangle hit group
    // A hit group specifies closest hit, any hit and intersection shaders to be executed when a ray intersects the geometry's triangle/AABB.
    // In this sample, we only use triangle geometry with a closest hit shader, so others are not set.
    auto hitGroup = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    hitGroup->SetClosestHitShaderImport(c_closestHitShaderName);
    hitGroup->SetHitGroupExport(c_hitGroupName);
    hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);

    // Shader config
    // Defines the maximum sizes in bytes for the ray payload and attribute structure.
    auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    UINT payloadSize = sizeof(RayPayload);
    UINT attributeSize = sizeof(XMFLOAT2);  // float2 barycentrics
    shaderConfig->Config(payloadSize, attributeSize);

    // Local root signature and shader association
    // This is a root signature that enables a shader to have unique arguments that come from shader tables.
    CreateLocalRootSignatureSubobjects(&raytracingPipeline);

    // Global root signature
    // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
    auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    globalRootSignature->SetRootSignature(m_raytracingGlobalRootSignature.Get());

    // Pipeline config
    // Defines the maximum TraceRay() recursion depth.
    auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
    // PERFOMANCE TIP: Set max recursion depth as low as needed 
    // as drivers may apply optimization strategies for low recursion depths.
    UINT maxRecursionDepth = 7; // ~ primary rays only. MSR_TODO
    pipelineConfig->Config(maxRecursionDepth);

#if _DEBUG
    PrintStateObjectDesc(raytracingPipeline);
#endif

    // Create the state object.
    ThrowIfFailed(m_dxrDevice->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_dxrStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
}

void PathTrace::CreateDescriptorHeap()
{
    auto device = m_deviceResources->GetD3DDevice();

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    // Allocate a heap for 6 descriptors:
    // 4 - vertex and index buffer SRVs
    // 1 - material SRV
    // 1 - raytracing output texture SRV
    descriptorHeapDesc.NumDescriptors = 6;
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    descriptorHeapDesc.NodeMask = 0;
    device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&m_descriptorHeap));
    NAME_D3D12_OBJECT(m_descriptorHeap);

    m_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

static void InterleaveArrays(size_t numArrays, uint8_t** srcArrays, size_t* attributeByteSize, size_t numAttributes, uint8_t* dest, size_t* outStride)
{
    size_t stride = 0;
    for (uint32_t i = 0; i < numArrays; i++)
    {
        stride += attributeByteSize[i];
    }

    size_t totalSize = stride * numAttributes;
    uint8_t* scratchBuf = new uint8_t[totalSize];
    uint8_t* curScratch = scratchBuf;

    for (uint32_t iAttrib = 0; iAttrib < numAttributes; iAttrib++)
    {
        for (uint32_t iArray = 0; iArray < numArrays; iArray++)
        {
            memcpy(curScratch, srcArrays[iArray] + iAttrib * attributeByteSize[iArray], attributeByteSize[iArray]);
            curScratch += attributeByteSize[iArray];
        }
    }

    memcpy(dest, scratchBuf, totalSize);

    delete[] scratchBuf;

    if (outStride != nullptr)
    {
        *outStride = stride;
    }
}

struct GltfModel
{
    tinygltf::Model model;
    size_t numVertices;
    size_t vertexStride;
    size_t verticesSize;
    const uint8_t* vertices;
    size_t numIndices;
    size_t indicesSize;
    const uint8_t* indices;
};

// Interleaves vertex data in place, destructively
void LoadGltf(const char* filename, GltfModel* dstModel)
{
    tinygltf::Model& model = dstModel->model;
    tinygltf::TinyGLTF loader;
    std::string error;
    std::string warning;

    loader.LoadBinaryFromFile(&model, &error, &warning, filename);
    auto& mesh = model.meshes[0];
    auto& primitive = mesh.primitives[0];

    int posAccessorIndex = -1;
    int normalAccessorIndex = -1;
    int tangentAccessorIndex = -1;
    for (const auto& attribute : primitive.attributes)
    {
        if (attribute.first == "POSITION")
        {
            posAccessorIndex = attribute.second;
        }
        else if (attribute.first == "NORMAL")
        {
            normalAccessorIndex = attribute.second;
        }
        else if (attribute.first == "TANGENT")
        {
            tangentAccessorIndex = attribute.second;
        }
    }

    assert(posAccessorIndex > -1);
    auto& posAccessor = model.accessors[posAccessorIndex];
    size_t vertexOffset = posAccessor.byteOffset;
    auto& posBufferView = model.bufferViews[posAccessor.bufferView];
    uint8_t* gltfVertices = model.buffers[posBufferView.buffer].data.data() + vertexOffset + posBufferView.byteOffset;

    assert(normalAccessorIndex > -1);
    auto& normalAccessor = model.accessors[normalAccessorIndex];
    size_t vertexNormalOffset = normalAccessor.byteOffset;
    auto& normalBufferView = model.bufferViews[normalAccessor.bufferView];
    uint8_t* gltfNormals = model.buffers[normalBufferView.buffer].data.data() + vertexNormalOffset + normalBufferView.byteOffset;

    assert(tangentAccessorIndex > -1);
    auto& tangentAccessor = model.accessors[tangentAccessorIndex];
    size_t vertexTangentOffset = tangentAccessor.byteOffset;
    auto& tangentBufferView = model.bufferViews[tangentAccessor.bufferView];
    uint8_t* gltfTangents = model.buffers[tangentBufferView.buffer].data.data() + vertexTangentOffset + tangentBufferView.byteOffset;

    uint8_t* streams[] = { gltfVertices, gltfNormals, gltfTangents };
    const size_t attribByteSize = sizeof(float) * 3;
    size_t attribByteSizes[] = { attribByteSize, attribByteSize, attribByteSize };

    size_t gltfVertexStride = 0;
    InterleaveArrays(3, streams, attribByteSizes, posAccessor.count, gltfVertices, &gltfVertexStride);
    size_t gltfVerticesSize = posAccessor.count * gltfVertexStride;

    auto& indexAccessorIndex = primitive.indices;
    auto& indexAccessor = model.accessors[indexAccessorIndex];
    size_t indexOffset = indexAccessor.byteOffset;

    assert(indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT);
    assert(indexAccessor.type == TINYGLTF_TYPE_SCALAR);

    auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
    uint8_t* gltfIndices = model.buffers[indexBufferView.buffer].data.data() + indexOffset + indexBufferView.byteOffset;
    size_t gltfIndicesSize = indexBufferView.byteLength;

    dstModel->numVertices = posAccessor.count;
    dstModel->vertexStride = gltfVertexStride;
    dstModel->verticesSize = gltfVerticesSize;
    dstModel->vertices = gltfVertices;
    dstModel->indicesSize = gltfIndicesSize;
    dstModel->indices = gltfIndices;
}

void PathTrace::BuildGeometry()
{
    GltfModel gltfInstancedModel;
    LoadGltf("Working/simple_sapling.glb", &gltfInstancedModel);

    GltfModel gltfOpenBoxModel;
    LoadGltf("Working/open_box.glb", &gltfOpenBoxModel);

    auto device = m_deviceResources->GetD3DDevice();

    const void* indexArrays[] = { gltfInstancedModel.indices, gltfOpenBoxModel.indices };
    size_t indexArraySizes[] = { gltfInstancedModel.indicesSize, gltfOpenBoxModel.indicesSize };

    const void* vertexArrays[] = { gltfInstancedModel.vertices, gltfOpenBoxModel.vertices };
    size_t vertexArraySizes[] = { gltfInstancedModel.verticesSize, gltfOpenBoxModel.verticesSize };
    size_t vertexArrayStrides[] = { gltfInstancedModel.vertexStride, gltfOpenBoxModel.vertexStride };

    for (int i = 0; i < ARRAYSIZE(indexArrays); ++i)
    {
        m_indexBuffers.emplace_back();
        AllocateUploadBuffer(device, indexArrays[i], indexArraySizes[i], &m_indexBuffers.back().resource);
        CreateBufferSRV(&m_indexBuffers.back(), (UINT)indexArraySizes[i] / 4, 0);
    }

    for (int i = 0; i < ARRAYSIZE(vertexArrays); ++i)
    {
        m_vertexBuffers.emplace_back();
        AllocateUploadBuffer(device, vertexArrays[i], vertexArraySizes[i], &m_vertexBuffers.back().resource);
        CreateBufferSRV(&m_vertexBuffers.back(), (UINT)(vertexArraySizes[i] / vertexArrayStrides[i]), (UINT)vertexArrayStrides[i]);
    }

    // Material information
    static float emissiveIntensity = 3.0f;
    Material materials[] =
    { 
        { { 0.0f, 0.0f, 0.0f, 0.0f }, { emissiveIntensity, emissiveIntensity, emissiveIntensity, 1.0f } },
        { { 0.5f, 0.5f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 1.0f } },
        { { 1.0f, 0.5f, 0.5f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
        { { 0.5f, 1.0f, 0.5f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } },
        { { 1.0f, 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } }
    };

    AllocateUploadBuffer(device, materials, sizeof(materials), &m_materialBuffer.resource);
    CreateBufferSRV(&m_materialBuffer, sizeof(materials) / sizeof(Material), sizeof(Material));
}

// MSR_TODO: Split out into alloc and build functions
static void BuildBlas(
    DX::DeviceResources& deviceResources,
    ComPtr<ID3D12Device5>& dxrDevice,
    ComPtr<ID3D12GraphicsCommandList5>& dxrCommandList,
    const std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>& geometryDescs,
    ComPtr<ID3D12Resource>& outScratchResource,
    ComPtr<ID3D12Resource>& outBlas)
{
    auto device = deviceResources.GetD3DDevice();

    // Get required sizes for an acceleration structure.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& bottomLevelInputs = bottomLevelBuildDesc.Inputs;
    bottomLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottomLevelInputs.Flags = buildFlags;
    bottomLevelInputs.NumDescs = (UINT)geometryDescs.size();
    bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomLevelInputs.pGeometryDescs = geometryDescs.data();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
    dxrDevice->GetRaytracingAccelerationStructurePrebuildInfo(&bottomLevelInputs, &bottomLevelPrebuildInfo);
    ThrowIfFalse(bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);

    AllocateUAVBuffer(device, bottomLevelPrebuildInfo.ScratchDataSizeInBytes, &outScratchResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"BlasScratchResource");

    // Allocate resources for acceleration structures.
    // Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
    // Default heap is OK since the application doesn’t need CPU read/write access to them. 
    // The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
    // and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
    //  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
    //  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.
    {
        D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;

        AllocateUAVBuffer(device, bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes, &outBlas, initialResourceState, L"BottomLevelAccelerationStructure");
    }

    // Bottom Level Acceleration Structure desc
    {
        bottomLevelBuildDesc.ScratchAccelerationStructureData = outScratchResource->GetGPUVirtualAddress();
        bottomLevelBuildDesc.DestAccelerationStructureData = outBlas->GetGPUVirtualAddress();
    }

    dxrCommandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);
}

void PathTrace::BuildAccelerationStructures()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    auto commandAllocator = m_deviceResources->GetCommandAllocator();

    // Reset the command list for the acceleration structure construction.
    commandList->Reset(commandAllocator, nullptr);

    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;
    std::vector<ComPtr<ID3D12Resource>> blasScratchResources;

    for (int i = 0; i < m_indexBuffers.size(); ++i)
    {
        geometryDescs.emplace_back();
        geometryDescs.back().Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometryDescs.back().Triangles.IndexBuffer = m_indexBuffers[i].resource->GetGPUVirtualAddress();
        geometryDescs.back().Triangles.IndexCount = static_cast<UINT>(m_indexBuffers[i].resource->GetDesc().Width) / sizeof(Index);
        geometryDescs.back().Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
        geometryDescs.back().Triangles.Transform3x4 = 0;
        geometryDescs.back().Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geometryDescs.back().Triangles.VertexCount = static_cast<UINT>(m_vertexBuffers[i].resource->GetDesc().Width) / sizeof(Vertex);
        geometryDescs.back().Triangles.VertexBuffer.StartAddress = m_vertexBuffers[i].resource->GetGPUVirtualAddress();
        geometryDescs.back().Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex); // TODO: FIXMEFIXMEFIXME

        // Mark the geometry as opaque. 
        // PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing optimizations.
        // Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is present or not.
        geometryDescs.back().Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

        ComPtr<ID3D12Resource> blasScratchResource;
        ComPtr<ID3D12Resource> bottomLevelAccelerationStructure;
        BuildBlas(
            *m_deviceResources.get(), 
            m_dxrDevice, 
            m_dxrCommandList, 
            std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>(geometryDescs.begin() + i, geometryDescs.begin() + i + 1), 
            blasScratchResource, 
            bottomLevelAccelerationStructure);

        blasScratchResources.emplace_back(blasScratchResource);
        m_bottomLevelAccelerationStructures.emplace_back(bottomLevelAccelerationStructure);
    }

    // Create instance descs for the bottom-level acceleration structures.
    ComPtr<ID3D12Resource> instanceDescBuffer;
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDescs;
    instanceDescs.emplace_back();
    instanceDescs.back().Transform[0][0] = instanceDescs.back().Transform[1][1] = instanceDescs.back().Transform[2][2] = 0.5f;
    instanceDescs.back().Transform[2][3] = -2.0f;
    instanceDescs.back().InstanceMask = 1;
    instanceDescs.back().InstanceID = 0;
    instanceDescs.back().AccelerationStructure = m_bottomLevelAccelerationStructures[0]->GetGPUVirtualAddress();
    instanceDescs.emplace_back();
    instanceDescs.back().Transform[0][0] = instanceDescs.back().Transform[1][1] = instanceDescs.back().Transform[2][2] = 0.5f;
    instanceDescs.back().Transform[2][3] = 2.0f;
    instanceDescs.back().InstanceMask = 1;
    instanceDescs.back().InstanceID = 0;
    instanceDescs.back().AccelerationStructure = m_bottomLevelAccelerationStructures[0]->GetGPUVirtualAddress();
    instanceDescs.emplace_back();
    instanceDescs.back().Transform[0][0] = instanceDescs.back().Transform[1][1] = instanceDescs.back().Transform[2][2] = 0.5f;
    instanceDescs.back().Transform[0][3] = 3.0f;
    instanceDescs.back().InstanceMask = 1;
    instanceDescs.back().InstanceID = 0;
    instanceDescs.back().AccelerationStructure = m_bottomLevelAccelerationStructures[0]->GetGPUVirtualAddress();
    instanceDescs.emplace_back();
    instanceDescs.back().Transform[0][0] = instanceDescs.back().Transform[1][1] = instanceDescs.back().Transform[2][2] = 0.5f;
    instanceDescs.back().Transform[0][3] = -3.0f;
    instanceDescs.back().InstanceMask = 1;
    instanceDescs.back().InstanceID = 0;
    instanceDescs.back().AccelerationStructure = m_bottomLevelAccelerationStructures[0]->GetGPUVirtualAddress();
    instanceDescs.emplace_back();
    instanceDescs.back().Transform[0][0] = instanceDescs.back().Transform[1][1] = instanceDescs.back().Transform[2][2] = 5.0f;
    instanceDescs.back().Transform[1][3] = 5.0f;
    instanceDescs.back().InstanceMask = 1;
    instanceDescs.back().InstanceID = 1;
    instanceDescs.back().AccelerationStructure = m_bottomLevelAccelerationStructures[1]->GetGPUVirtualAddress();

    AllocateUploadBuffer(device, instanceDescs.data(), instanceDescs.size() * sizeof(instanceDescs[0]), &instanceDescBuffer, L"InstanceDescBuffer");

    // Get required sizes for an acceleration structure.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& topLevelInputs = topLevelBuildDesc.Inputs;
    topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    topLevelInputs.Flags = buildFlags;
    topLevelInputs.NumDescs = (UINT)instanceDescs.size();
    topLevelInputs.pGeometryDescs = nullptr;
    topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
    m_dxrDevice->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);
    ThrowIfFalse(topLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0);

    ComPtr<ID3D12Resource> scratchResource;
    AllocateUAVBuffer(device, topLevelPrebuildInfo.ScratchDataSizeInBytes, &scratchResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"TlasScratchResource");

    // Allocate resources for acceleration structures.
    // Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
    // Default heap is OK since the application doesn’t need CPU read/write access to them. 
    // The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
    // and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
    //  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
    //  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.
    {
        D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        AllocateUAVBuffer(device, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, &m_topLevelAccelerationStructure, initialResourceState, L"TopLevelAccelerationStructure");
    }

    // Top Level Acceleration Structure desc
    {
        topLevelBuildDesc.DestAccelerationStructureData = m_topLevelAccelerationStructure->GetGPUVirtualAddress();
        topLevelBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
        topLevelBuildDesc.Inputs.InstanceDescs = instanceDescBuffer->GetGPUVirtualAddress();
    }

    {
        // Build top level acceleration structure.
        CD3DX12_RESOURCE_BARRIER blasBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_bottomLevelAccelerationStructures[0].Get());
        commandList->ResourceBarrier(1, &blasBarrier);
        m_dxrCommandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);
    };

    // Kick off acceleration structure construction.
    m_deviceResources->ExecuteCommandList();

    // Wait for GPU to finish as the locally created temporary GPU resources will get released once we go out of scope.
    m_deviceResources->WaitForGpu();
}

void PathTrace::CreateConstantBuffers()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto frameCount = m_deviceResources->GetBackBufferCount();

    // Create the constant buffer memory and map the CPU and GPU addresses
    const D3D12_HEAP_PROPERTIES uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    // Allocate one constant buffer per frame, since it gets updated every frame.
    size_t cbSize = frameCount * sizeof(AlignedSceneConstantBuffer);
    const D3D12_RESOURCE_DESC constantBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(cbSize);

    ThrowIfFailed(device->CreateCommittedResource(
        &uploadHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &constantBufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_perFrameConstants)));

    // Map the constant buffer and cache its heap pointers.
    // We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay.
    CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
    ThrowIfFailed(m_perFrameConstants->Map(0, nullptr, reinterpret_cast<void**>(&m_mappedConstantData)));
}

void PathTrace::BuildShaderTables()
{
    auto device = m_deviceResources->GetD3DDevice();

    void* rayGenShaderIdentifier;
    void* missShaderIdentifier;
    void* hitGroupShaderIdentifier;

    auto GetShaderIdentifiers = [&](auto* stateObjectProperties)
    {
        rayGenShaderIdentifier = stateObjectProperties->GetShaderIdentifier(c_raygenShaderName);
        missShaderIdentifier = stateObjectProperties->GetShaderIdentifier(c_missShaderName);
        hitGroupShaderIdentifier = stateObjectProperties->GetShaderIdentifier(c_hitGroupName);
    };

    // Get shader identifiers.
    UINT shaderIdentifierSize;
    {
        ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
        ThrowIfFailed(m_dxrStateObject.As(&stateObjectProperties));
        GetShaderIdentifiers(stateObjectProperties.Get());
        shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    }

    // Ray gen shader table
    {
        UINT numShaderRecords = 1;
        UINT shaderRecordSize = shaderIdentifierSize;
        ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable");
        rayGenShaderTable.push_back(ShaderRecord(rayGenShaderIdentifier, shaderIdentifierSize));
        m_rayGenShaderTable = rayGenShaderTable.GetResource();
    }

    // Miss shader table
    {
        UINT numShaderRecords = 1;
        UINT shaderRecordSize = shaderIdentifierSize;
        ShaderTable missShaderTable(device, numShaderRecords, shaderRecordSize, L"MissShaderTable");
        missShaderTable.push_back(ShaderRecord(missShaderIdentifier, shaderIdentifierSize));
        m_missShaderTable = missShaderTable.GetResource();
    }

    // Hit group shader table
    {
        struct RootArguments {
            CubeConstantBuffer cb;
        } rootArguments;
        rootArguments.cb = m_cubeCB;

        UINT numShaderRecords = 1;
        UINT shaderRecordSize = shaderIdentifierSize + sizeof(rootArguments);
        ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");
        hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderIdentifier, shaderIdentifierSize, &rootArguments, sizeof(rootArguments)));
        m_hitGroupShaderTable = hitGroupShaderTable.GetResource();
    }
}

void PathTrace::CreateRaytracingOutputResource()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

    // Create the output resource. The dimensions and format should match the swap-chain.
    auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(backbufferFormat, m_width, m_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(device->CreateCommittedResource(
        &defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &uavDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_raytracingOutput)));
    NAME_D3D12_OBJECT(m_raytracingOutput);

    D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
    m_raytracingOutputResourceUAVDescriptorHeapIndex = AllocateDescriptor(&uavDescriptorHandle, m_raytracingOutputResourceUAVDescriptorHeapIndex);
    D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
    UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(m_raytracingOutput.Get(), nullptr, &UAVDesc, uavDescriptorHandle);
    m_raytracingOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), m_raytracingOutputResourceUAVDescriptorHeapIndex, m_descriptorSize);
}

// Allocate a descriptor and return its index. 
// If the passed descriptorIndexToUse is valid, it will be used instead of allocating a new one.
UINT PathTrace::AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse)
{
    auto descriptorHeapCpuBase = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    if (descriptorIndexToUse >= m_descriptorHeap->GetDesc().NumDescriptors)
    {
        descriptorIndexToUse = m_descriptorsAllocated++;
    }
    *cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeapCpuBase, descriptorIndexToUse, m_descriptorSize);
    return descriptorIndexToUse;
}

// Create SRV for a buffer.
UINT PathTrace::CreateBufferSRV(D3DBuffer* buffer, UINT numElements, UINT elementSize)
{
    auto device = m_deviceResources->GetD3DDevice();

    // SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.NumElements = numElements;
    if (elementSize == 0)
    {
        srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        srvDesc.Buffer.StructureByteStride = 0;
    }
    else
    {
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        srvDesc.Buffer.StructureByteStride = elementSize;
    }
    UINT descriptorIndex = AllocateDescriptor(&buffer->cpuDescriptorHandle);
    device->CreateShaderResourceView(buffer->resource.Get(), &srvDesc, buffer->cpuDescriptorHandle);
    buffer->gpuDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_descriptorHeap->GetGPUDescriptorHandleForHeapStart(), descriptorIndex, m_descriptorSize);
    return descriptorIndex;
}

void PathTrace::DoRaytracing()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    auto DispatchRays = [&](auto* commandList, auto* stateObject, auto* dispatchDesc)
    {
        // Since each shader table has only one shader record, the stride is same as the size.
        dispatchDesc->HitGroupTable.StartAddress = m_hitGroupShaderTable->GetGPUVirtualAddress();
        dispatchDesc->HitGroupTable.SizeInBytes = m_hitGroupShaderTable->GetDesc().Width;
        dispatchDesc->HitGroupTable.StrideInBytes = dispatchDesc->HitGroupTable.SizeInBytes;
        dispatchDesc->MissShaderTable.StartAddress = m_missShaderTable->GetGPUVirtualAddress();
        dispatchDesc->MissShaderTable.SizeInBytes = m_missShaderTable->GetDesc().Width;
        dispatchDesc->MissShaderTable.StrideInBytes = dispatchDesc->MissShaderTable.SizeInBytes;
        dispatchDesc->RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGPUVirtualAddress();
        dispatchDesc->RayGenerationShaderRecord.SizeInBytes = m_rayGenShaderTable->GetDesc().Width;
        dispatchDesc->Width = m_width;
        dispatchDesc->Height = m_height;
        dispatchDesc->Depth = 1;
        commandList->SetPipelineState1(stateObject);
        commandList->DispatchRays(dispatchDesc);
    };

    auto SetCommonPipelineState = [&](auto* descriptorSetCommandList)
    {
        descriptorSetCommandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
        // Set index and successive vertex buffer decriptor tables
        commandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::VertexBuffersSlot, m_indexBuffers[0].gpuDescriptorHandle);
        commandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::OutputViewSlot, m_raytracingOutputResourceUAVGpuDescriptor);
    };

    commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

    // Copy the updated scene constant buffer to GPU.
    memcpy(&m_mappedConstantData[frameIndex].constants, &m_sceneCB[frameIndex], sizeof(m_sceneCB[frameIndex]));
    auto cbGpuAddress = m_perFrameConstants->GetGPUVirtualAddress() + frameIndex * sizeof(m_mappedConstantData[0]);
    commandList->SetComputeRootConstantBufferView(GlobalRootSignatureParams::SceneConstantSlot, cbGpuAddress);

    // Bind the heaps, acceleration structure and dispatch rays.
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    SetCommonPipelineState(commandList);
    commandList->SetComputeRootShaderResourceView(GlobalRootSignatureParams::AccelerationStructureSlot, m_topLevelAccelerationStructure->GetGPUVirtualAddress());
    DispatchRays(m_dxrCommandList.Get(), m_dxrStateObject.Get(), &dispatchDesc);
}

void PathTrace::CopyRaytracingOutputToBackbuffer()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto renderTarget = m_deviceResources->GetRenderTarget();

    D3D12_RESOURCE_BARRIER preCopyBarriers[2];
    preCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
    preCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    commandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

    commandList->CopyResource(renderTarget, m_raytracingOutput.Get());

    D3D12_RESOURCE_BARRIER postCopyBarriers[2];
    postCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
    postCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutput.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    commandList->ResourceBarrier(ARRAYSIZE(postCopyBarriers), postCopyBarriers);
}

void PathTrace::OnKeyDown(UINT8 key)
{
    const float rotationScale = 0.05f;
    const float forwardScale = 0.1f;

    switch (key)
    {
    case VK_ESCAPE:
        m_deviceResources->WaitForGpu();
        PostQuitMessage(0);
        break;
    case VK_LEFT:
        UpdateRotation(0.0f, XM_PI * rotationScale);
        break;
    case VK_RIGHT:
        UpdateRotation(0.0f, -XM_PI * rotationScale);
        break;
    case VK_UP:
        UpdateRotation(XM_PI * rotationScale, 0.0f);
        break;
    case VK_DOWN:
        UpdateRotation(-XM_PI * rotationScale, 0.0f);
        break;
    case VK_SPACE:
        UpdatePosition(forwardScale);
        break;
    case VK_SHIFT:
        UpdatePosition(-forwardScale);
        break;
    default: break;
    }
}
