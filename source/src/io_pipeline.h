#ifndef ASEVOXEL_IO_PIPELINE_H
#define ASEVOXEL_IO_PIPELINE_H

#include <cstdint>
#include <vector>
#include <string>

// ============================================================================
// Binary protocol data structures
// Must match kernel.h OpenCL structs exactly (packed, little-endian)
// ============================================================================

#pragma pack(push, 1)

struct Voxel {
    float    x, y, z;   // 12 bytes: world position
    uint32_t rgba;       //  4 bytes: packed RGBA (R in low byte)
};                       // 16 bytes total

struct RenderParams {
    uint32_t width, height;     //  8 bytes
    uint32_t voxelCount;        //  4 bytes
    uint32_t flags;             //  4 bytes (bit0: perspective, bit1: lighting, bit2: backface cull,
                                //           bit3: dynamic lighting, bit4: rim,
                                //           bits5-6: raster mode 0=splat, 1=slice, 2=mesh)
    float    rot[9];            // 36 bytes: 3×3 row-major rotation matrix
    float    middle[3];         // 12 bytes: rotation pivot
    float    voxelSize;         //  4 bytes: pixels per unit
    float    fovDegrees;        //  4 bytes: FOV for perspective mode
    float    cameraZ;           //  4 bytes: camera Z position
    float    centerX, centerY;  //  8 bytes: screen center
    float    lightIntensity;    //  4 bytes: 0-100 (basic mode: bright side)
    float    shadeIntensity;    //  4 bytes: 0-100 (basic mode: dark side)
    float    pad[9];            // 36 bytes: dynamic-lighting payload
                                //   pad[0]: previewVoxelCount (uint32 reinterpreted)
                                //   pad[1]: dynamic pitch   (-90..90 degrees)
                                //   pad[2]: dynamic yaw     (0..360 degrees)
                                //   pad[3]: dynamic diffuse (0..100)
                                //   pad[4]: dynamic ambient (0..100)
                                //   pad[5]: dynamic coneAngle (0=cylinder,90=cone,180=plane)
                                //   pad[6]: dynamic diameter (0..110, relative to model size)
                                //   pad[7]: dynamic lightColor (uint32 RGBA reinterpreted)
                                //   pad[8]: modelRadius (bounding-sphere radius, radial falloff)
};                              // 128 bytes total

// Extended FX shader params — optional, appended after voxels in the IPC stream.
// Zeros = no FX active (used as default when absent from the stream).
// Supports up to 8 material-scoped faceshade slots and 4 iso slots.
// RGBA packing: R in low byte. alpha-mode: A=brightness, RGB=tint. literal: full RGBA.
// materialColor: RGB-matched against voxel color (alpha ignored); 0 = global scope.

// Global flags (ExtShaderParams.flags)
#define EXT_FLAG_BLUR            64u   // screen-space blur post-process
#define EXT_FLAG_BLUR_DEPTH      128u  // depth-aware blur (gates across silhouettes)

// Per-slot flags (FaceshadeSlot.slotFlags / IsoSlot.slotFlags)
#define EXT_SLOT_LIT             1u    // literal mode (replace color entirely)
#define EXT_SLOT_TINT            2u    // alpha + tint (alpha mode only)

// Per-faceshade slot: 32 bytes
struct FaceshadeSlot {
    uint32_t materialColor;  //  4: RGBA match key; 0 = global (any voxel)
    uint32_t slotFlags;      //  4: EXT_SLOT_LIT, EXT_SLOT_TINT
    uint32_t face[6];        // 24: per-face RGBA (front,back,right,left,top,bottom)
};                           // 32 bytes

// Per-iso slot: 24 bytes
struct IsoSlot {
    uint32_t materialColor;  //  4: RGBA match key; 0 = global
    uint32_t slotFlags;      //  4: EXT_SLOT_LIT, EXT_SLOT_TINT
    uint32_t iso[3];         // 12: RGBA (top=y-dominant, left=x<0, right=x>=0)
    uint32_t pad;            //  4: alignment
};                           // 24 bytes

struct ExtShaderParams {
    uint32_t      flags;          //  4: EXT_FLAG_BLUR, EXT_FLAG_BLUR_DEPTH
    uint32_t      numFaceshade;   //  4: active FaceshadeSlot count (0..8)
    uint32_t      numIso;         //  4: active IsoSlot count (0..4)
    uint32_t      headerPad;      //  4: reserved, must be zero
    FaceshadeSlot faceshades[8];  // 256: faceshade slots
    IsoSlot       isos[4];        //  96: iso slots
    uint32_t      blurFlags;      //  4: (reserved)
    uint32_t      blurRadius;     //  4: 1..8
    uint32_t      blurStrength;   //  4: 0..100
    uint32_t      blurKernelType; //  4: 0=gaussian, 1=box, 2=triangle
};                                // 16 + 256 + 96 + 16 = 384 bytes total

#pragma pack(pop)

static_assert(sizeof(Voxel) == 16, "Voxel must be 16 bytes");
static_assert(sizeof(RenderParams) == 128, "RenderParams must be 128 bytes");
static_assert(sizeof(FaceshadeSlot) == 32, "FaceshadeSlot must be 32 bytes");
static_assert(sizeof(IsoSlot) == 24, "IsoSlot must be 24 bytes");
static_assert(sizeof(ExtShaderParams) == 384, "ExtShaderParams must be 384 bytes");

// ============================================================================
// I/O functions
// ============================================================================

struct RenderInput {
    RenderParams           params;
    std::vector<Voxel>     voxels;
    ExtShaderParams        extShaderParams = {};  // zero-init = no FX active
};

// Read RenderParams + Voxel array from file or stdin.
// Returns 0 on success, EXIT_IO_ERROR on failure.
int read_input(const std::string& path, RenderInput& input);
int read_input_stdin(RenderInput& input);

// Write width, height, RGBA pixel data to stdout.
// Returns 0 on success, EXIT_IO_ERROR on failure.
int write_output(uint32_t width, uint32_t height, const uint8_t* pixels);

#endif // ASEVOXEL_IO_PIPELINE_H
