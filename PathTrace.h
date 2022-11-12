// PathTrace.h

#pragma once

#include "DXSample.h"
#include "Shaders/SharedDefinitions.hlsli"

namespace GlobalRootSignatureParams 
{
    enum Value 
    {
        OutputViewSlot = 0,
        AccelerationStructureSlot,
        SceneConstantSlot,
        VertexBuffersSlot,
        Count
    };
}

namespace LocalRootSignatureParams 
{
    enum Value 
    {
        CubeConstantSlot = 0,
        Count
    };
}

class PathTrace : public DXSample
{
public:
    PathTrace(UINT width, UINT height, std::wstring name);

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // DXSample
    virtual void OnInit() override;
    virtual void OnUpdate() override;
    virtual void OnRender() override;
    virtual void OnSizeChanged(UINT width, UINT height, bool minimized) override;
    virtual void OnDestroy() override;
    virtual void OnKeyDown(UINT8 key) override;

private:
    static const UINT FrameCount = 3;

    // We'll allocate space for several of these and they will need to be padded for alignment.
    static_assert(sizeof(SceneConstantBuffer) < D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, "Checking the size here.");

    union AlignedSceneConstantBuffer
    {
        SceneConstantBuffer constants;
        uint8_t             alignmentPadding[D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT];
    };
    AlignedSceneConstantBuffer*  m_mappedConstantData;
    ComPtr<ID3D12Resource>       m_perFrameConstants;

    // DirectX Raytracing (DXR) attributes
    ComPtr<ID3D12Device5>              m_dxrDevice;
    ComPtr<ID3D12GraphicsCommandList5> m_dxrCommandList;
    ComPtr<ID3D12StateObject>          m_dxrStateObject;

    // Root signatures
    ComPtr<ID3D12RootSignature> m_raytracingGlobalRootSignature;
    ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature;

    // Descriptors
    ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
    UINT                         m_descriptorsAllocated;
    UINT                         m_descriptorSize;

    // Raytracing scene
    SceneConstantBuffer m_sceneCB[FrameCount];
    CubeConstantBuffer  m_cubeCB;

    // Geometry
    struct D3DBuffer
    {
        ComPtr<ID3D12Resource>      resource;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle;
        D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorHandle;
    };

    std::vector<D3DBuffer> m_indexBuffers;
    std::vector<D3DBuffer> m_vertexBuffers;
    D3DBuffer              m_materialBuffer;

    // Acceleration structure
    //ComPtr<ID3D12Resource> m_bottomLevelAccelerationStructure;
    std::vector<ComPtr<ID3D12Resource>> m_bottomLevelAccelerationStructures;
    ComPtr<ID3D12Resource>              m_topLevelAccelerationStructure;

    // Raytracing output
    ComPtr<ID3D12Resource>      m_raytracingOutput;
    D3D12_GPU_DESCRIPTOR_HANDLE m_raytracingOutputResourceUAVGpuDescriptor;
    UINT                        m_raytracingOutputResourceUAVDescriptorHeapIndex;

    // Shader tables
    static const wchar_t*  c_hitGroupName;
    static const wchar_t*  c_raygenShaderName;
    static const wchar_t*  c_closestHitShaderName;
    static const wchar_t*  c_missShaderName;
    ComPtr<ID3D12Resource> m_missShaderTable;
    ComPtr<ID3D12Resource> m_hitGroupShaderTable;
    ComPtr<ID3D12Resource> m_rayGenShaderTable;

    // Application state
    float             m_curRotationAngleRad;
    DirectX::XMVECTOR m_eye;
    DirectX::XMVECTOR m_at;
    DirectX::XMVECTOR m_up;
    int               m_frameNumber = 0;

    void CreateDeviceDependentResources();
    void ReleaseDeviceDependentResources();
    void CreateWindowSizeDependentResources();
    void ReleaseWindowSizeDependentResources();
    void CreateRaytracingInterfaces();
    void CreateRootSignatures();
    void CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateRaytracingPipelineStateObject();
    void CreateDescriptorHeap();
    void BuildGeometry();
    void BuildAccelerationStructures();
    void CreateConstantBuffers();
    void BuildShaderTables();
    void CreateRaytracingOutputResource();
    void SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig);
    void UpdateCameraMatrices();
    void InitializeScene();
    UINT AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse = UINT_MAX);
    UINT CreateBufferSRV(D3DBuffer* buffer, UINT numElements, UINT elementSize);
    void DoRaytracing();
    void CopyRaytracingOutputToBackbuffer();
    void UpdateRotation(float deltaX, float deltaY);
    void UpdatePosition(float forward);
};
