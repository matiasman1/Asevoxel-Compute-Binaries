#ifndef ASEVOXEL_KERNEL_H
#define ASEVOXEL_KERNEL_H

// ============================================================================
// Embedded OpenCL C kernel source for AseVoxel voxel rendering
// ============================================================================
// Parallelize over voxels (not pixels). Each work item transforms one voxel,
// projects it, performs atomic depth test, and writes to framebuffer.
// Matches the existing AseVoxel math conventions:
//   - 3×3 row-major rotation, Z·Y·X order (pre-computed on host)
//   - Orthographic OR perspective projection (flag-selected)
//   - Face shading with configurable exponent curve
//   - Atomic CAS depth buffer (integer-reinterpreted float)
// ============================================================================

static constexpr const char* KERNEL_SOURCE = R"CL(

// ---- Data structures (must match host-side layout exactly) -----------------

typedef struct __attribute__((packed)) {
    float x, y, z;       // world position
    uint  rgba;           // packed RGBA (R in low byte)
} Voxel;

typedef struct __attribute__((packed)) {
    uint  width, height;
    uint  voxelCount;
    uint  flags;             // bit0: perspective, bit1: lighting, bit2: backface cull,
                             // bit3: dynamic lighting model, bit4: rim lighting
    float rot[9];            // 3x3 row-major rotation matrix
    float middle[3];         // rotation pivot (model center)
    float voxelSize;         // pixels per unit
    float fovDegrees;        // FOV for perspective mode
    float cameraZ;           // camera Z position
    float centerX, centerY;  // screen center in pixels
    float lightIntensity;    // 0-100 (basic mode: bright side)
    float shadeIntensity;    // 0-100 (basic mode: dark side)
    float pad[9];            // padding to 128 bytes. Dynamic-lighting payload:
                             //   pad[0]: previewVoxelCount (uint32 reinterpreted)
                             //   pad[1]: dynamic pitch   (-90..90 degrees)
                             //   pad[2]: dynamic yaw     (0..360 degrees)
                             //   pad[3]: dynamic diffuse (0..100)
                             //   pad[4]: dynamic ambient (0..100)
                             //   pad[5]: dynamic coneAngle (0=cylinder,90=cone,180=plane)
                             //   pad[6]: dynamic diameter (0..110, relative to model size)
                             //   pad[7]: dynamic lightColor (uint32 RGBA reinterpreted)
                             //   pad[8]: modelRadius (bounding-sphere radius, for radial falloff)
} RenderParams;

// RenderParams flag bit helpers
#define FLAG_PERSPECTIVE   1u
#define FLAG_LIGHTING      2u
#define FLAG_BACKFACE      4u
#define FLAG_DYNAMIC       8u
#define FLAG_RIM           16u
// Rasterization mode (bits 5-6); 0=splat/rect (default), 1=slice, 2=mesh
#define FLAG_RASTER_SLICE  32u   // bit 5: render dominant face as projected quad
#define FLAG_RASTER_MESH   64u   // bit 6: render all visible faces as triangle mesh
#define FLAG_RASTER_MASK   96u   // bits 5-6

// ---- Extended FX shader params (384 bytes, optional buffer) ----------------
// Appended after voxel data in the IPC stream; zeros = no FX shaders active.
// Supports up to 8 material-scoped faceshade slots and 4 iso slots.
// RGBA packing: R in low byte (matches voxel RGBA convention)
//   alpha mode:   .a = brightness multiplier 0..255, .rgb = tint color
//   literal mode: full RGBA replaces the shaded color
//
// materialColor: RGBA to match against voxel's packed rgba (RGB compare only).
//   0 = global scope (applies to all voxels regardless of color).

// Global flags (ExtShaderParams.flags)
#define EXT_FLAG_BLUR            64u   // screen-space blur post-process
#define EXT_FLAG_BLUR_DEPTH      128u  // depth-aware blur (gates across silhouettes)

// Per-slot flags (FaceshadeSlot.slotFlags / IsoSlot.slotFlags)
#define EXT_SLOT_LIT             1u    // literal mode (replace color entirely)
#define EXT_SLOT_TINT            2u    // alpha + tint (alpha mode only)

// Per-faceshade slot: 32 bytes
// face[] order: 0=front(z+),1=back(z-),2=right(x+),3=left(x-),4=top(y+),5=bottom(y-)
typedef struct __attribute__((packed)) {
    uint  materialColor;  //  4: RGBA to match voxel; 0 = global (any voxel)
    uint  slotFlags;      //  4: EXT_SLOT_LIT, EXT_SLOT_TINT
    uint  face[6];        // 24: per-face RGBA colors
} FaceshadeSlot;          // 32 bytes

// Per-iso slot: 24 bytes
// iso[] order: 0=top(y-dominant), 1=left(x<0), 2=right(x>=0/z-dom)
typedef struct __attribute__((packed)) {
    uint  materialColor;  //  4: RGBA to match voxel; 0 = global
    uint  slotFlags;      //  4: EXT_SLOT_LIT, EXT_SLOT_TINT
    uint  iso[3];         // 12: per-iso-role RGBA colors
    uint  pad;            //  4: alignment padding
} IsoSlot;                // 24 bytes

typedef struct __attribute__((packed)) {
    uint          flags;          //  4: global flags (EXT_FLAG_BLUR, EXT_FLAG_BLUR_DEPTH)
    uint          numFaceshade;   //  4: active FaceshadeSlot count (0..8)
    uint          numIso;         //  4: active IsoSlot count (0..4)
    uint          headerPad;      //  4: reserved, must be zero
    FaceshadeSlot faceshades[8];  // 256: faceshade slots
    IsoSlot       isos[4];        //  96: iso slots
    uint          blurFlags;      //  4: (reserved, blur controlled by global flags)
    uint          blurRadius;     //  4: 1..8
    uint          blurStrength;   //  4: 0..100
    uint          blurKernelType; //  4: 0=gaussian, 1=box, 2=triangle
} ExtShaderParams;                // 16 + 256 + 96 + 16 = 384 bytes total

// ---- Helpers ---------------------------------------------------------------

// Rotate point around pivot by 3x3 row-major matrix
inline float3 rotate_point(float3 p, float3 pivot,
                           __global const float* rot) {
    float3 d = p - pivot;
    float3 r;
    r.x = rot[0]*d.x + rot[1]*d.y + rot[2]*d.z;
    r.y = rot[3]*d.x + rot[4]*d.y + rot[5]*d.z;
    r.z = rot[6]*d.x + rot[7]*d.y + rot[8]*d.z;
    return r + pivot;
}

// Rotate a direction vector (no pivot)
inline float3 rotate_dir(__global const float* rot, float3 d) {
    float3 r;
    r.x = rot[0]*d.x + rot[1]*d.y + rot[2]*d.z;
    r.y = rot[3]*d.x + rot[4]*d.y + rot[5]*d.z;
    r.z = rot[6]*d.x + rot[7]*d.y + rot[8]*d.z;
    return r;
}

// Project rotated world-space point to screen coordinates
// Returns (screenX, screenY, depth)
// Project using offset from rotation pivot (middle), matching the native
// renderer convention: screenPos = center + (worldPos - middle) * voxelSize.
inline float3 project(float3 worldPos,
                      __global const RenderParams* p) {
    float3 result;
    bool perspective = (p->flags & 1u) != 0;
    float relX = worldPos.x - p->middle[0];
    float relY = worldPos.y - p->middle[1];

    if (perspective) {
        float depth = p->cameraZ - worldPos.z;
        float fovRad = p->fovDegrees * 0.0174532925f * 0.5f;
        float focalLen = ((float)p->height * 0.5f) / tan(fovRad);
        float scale = focalLen / fmax(0.001f, depth);
        result.x = p->centerX + relX * p->voxelSize * scale;
        result.y = p->centerY + relY * p->voxelSize * scale;
        result.z = depth;
    } else {
        result.x = p->centerX + relX * p->voxelSize;
        result.y = p->centerY + relY * p->voxelSize;
        result.z = p->cameraZ - worldPos.z;
    }
    return result;
}

// Basic camera-facing shading — mirrors basic.lua formula exactly:
//   t = (dot + 1) / 2  →  brightness = shade + (light - shade) * t
// Returns a multiplier in [0, 1].
inline float shade_basic(float3 domN, float3 viewDir,
                          float lightInt, float shadeInt) {
    float dotVal = dot(domN, viewDir);
    float t = (dotVal + 1.0f) * 0.5f;
    float brightness = shadeInt + (lightInt - shadeInt) * t;
    return clamp(brightness * 0.01f, 0.0f, 1.0f);
}

// ---- Dynamic lighting helpers (mirror dynamic.cpp v1.3.0) ------------------

// Light direction from yaw/pitch (view-space, matches dynamic.cpp)
inline float3 compute_light_dir(float yawDeg, float pitchDeg) {
    float yr = yawDeg * 0.0174532925f;
    float pr = pitchDeg * 0.0174532925f;
    float cy = cos(yr), sy = sin(yr);
    float cp = cos(pr), sp = sin(pr);
    return (float3)(cy * cp, sp, sy * cp);
}

// Transform a view-space direction to model space using inverse rotation
// (inverse of a pure rotation is its transpose). rot is row-major.
inline float3 view_to_model_dir(__global const float* rot, float3 v) {
    float3 r;
    r.x = rot[0]*v.x + rot[3]*v.y + rot[6]*v.z;
    r.y = rot[1]*v.x + rot[4]*v.y + rot[7]*v.z;
    r.z = rot[2]*v.x + rot[5]*v.y + rot[8]*v.z;
    return r;
}

// Full dynamic-lighting evaluation for one (dominant) face.
// Returns the per-channel color multiplier (ambient + diffuse*lightColor) via
// colorMul, and the additive rim contribution (0-255 space) via rimAdd.
//   domN       : dominant rotated (view-space) face normal
//   mrel       : model-space voxel position relative to model center
//   p          : render params (carries dynamic payload in pad[1..8])
inline void shade_dynamic(float3 domN, float3 mrel,
                          __global const RenderParams* p,
                          float3* colorMul, float3* rimAdd) {
    float pitch      = p->pad[1];
    float yaw        = p->pad[2];
    float diffuseI   = p->pad[3] * 0.01f;
    float ambientI   = p->pad[4] * 0.01f;
    float coneAngle  = p->pad[5];
    float diameter   = p->pad[6];
    uint  lc         = as_uint(p->pad[7]);
    float modelRadius = p->pad[8];

    float lr = (float)(lc & 0xFFu)         / 255.0f;
    float lg = (float)((lc >> 8) & 0xFFu)  / 255.0f;
    float lb = (float)((lc >> 16) & 0xFFu) / 255.0f;

    float3 L = compute_light_dir(yaw, pitch);
    float ndotl = dot(domN, L);

    // Higher diffuse intensity => sharper Lambert falloff (range 1..4)
    float exponent = 1.0f + (1.0f - diffuseI) * 3.0f;
    float diffuse = 0.0f;

    if (coneAngle >= 179.0f) {
        // PLANE mode: uniform light on the lit hemisphere, no radial falloff
        diffuse = (ndotl > 0.0f) ? 1.0f : 0.0f;
    } else {
        float nl = ndotl < 0.0f ? 0.0f : ndotl;
        diffuse = pow(nl, exponent);

        float dia = diameter * 0.01f;
        if (dia < 0.0f) dia = 0.0f;
        float baseRadius = dia * modelRadius;
        if (baseRadius > 1e-6f) {
            // Radial attenuation uses model-space positions + model-space light dir
            float3 mL = view_to_model_dir(p->rot, L);
            float proj = dot(mrel, mL);
            float3 perp = mrel - proj * mL;
            float perpDist = length(perp);
            if (coneAngle <= 1.0f) {
                // CYLINDER: hard cutoff at radius, no interior falloff
                if (perpDist > baseRadius) diffuse = 0.0f;
            } else {
                // CONE: smooth falloff, sharper for narrower cones
                float nd = perpDist / baseRadius;
                float falloffPower = 0.5f + ((180.0f - coneAngle) / 180.0f) * 1.5f;
                float radial = 1.0f - pow(nd, falloffPower);
                if (radial < 0.0f) radial = 0.0f;
                diffuse *= radial;
            }
        }
    }

    diffuse *= diffuseI;

    *colorMul = (float3)(ambientI + diffuse * lr,
                         ambientI + diffuse * lg,
                         ambientI + diffuse * lb);

    // Rim lighting (Fresnel at silhouette). View dir is +Z (toward camera) in
    // kernel convention, so camera-facing normals have ndotv≈1 (edge≈0 → no rim).
    float3 rim = (float3)(0.0f, 0.0f, 0.0f);
    if ((p->flags & FLAG_RIM) != 0) {
        float ndotv = domN.z;  // dot(domN, (0,0,1))
        if (ndotv > 0.0f) {
            float edge = 1.0f - ndotv;
            float t = smoothstep(0.55f, 0.95f, edge);
            if (t > 0.0f) {
                float rs = 0.6f * t;
                rim = (float3)(lr * rs * 255.0f, lg * rs * 255.0f, lb * rs * 255.0f);
            }
        }
    }
    *rimAdd = rim;
}

// ---- FX shader helpers (faceshade, iso) ------------------------------------

// Apply a packed RGBA color entry to float r/g/b (0-255 range) and uint a.
//   mode 0 = alpha-only:  multiply r,g,b by entry.a/255
//   mode 1 = alpha+tint:  lerp(col*a, tint*a, 1-a)  → fade toward tint color
//   mode 2 = literal:     replace r,g,b,a with entry's RGBA
inline void apply_colorentry(uint entry, uint mode,
                              float* r, float* g, float* b, uint* a) {
    float er = (float)(entry & 0xFFu);
    float eg = (float)((entry >> 8)  & 0xFFu);
    float eb = (float)((entry >> 16) & 0xFFu);
    float ea = (float)((entry >> 24) & 0xFFu) / 255.0f;

    if (mode == 2u) {           // literal: replace color
        *r = er; *g = eg; *b = eb;
        *a = (entry >> 24) & 0xFFu;
    } else if (mode == 1u) {    // alpha + tint: mirrors faceshade.lua tint path
        float tintAmt = 1.0f - ea;
        *r = (*r) * ea * ea + er * ea * tintAmt;
        *g = (*g) * ea * ea + eg * ea * tintAmt;
        *b = (*b) * ea * ea + eb * ea * tintAmt;
    } else {                    // alpha only: multiply brightness
        *r *= ea;
        *g *= ea;
        *b *= ea;
    }
}

// Apply the full FX shader chain (faceshade + iso slots) for ONE specific
// face, given that face's index (matches FACE_NORMALS[]/FaceshadeSlot.face[]
// order: 0=front,1=back,2=right,3=left,4=top,5=bottom) and its rotated
// (view-space) normal, used for iso role classification.
inline void apply_fx_for_face(
    __global const ExtShaderParams* extParams, uint voxelRGB,
    int faceIdx, float3 faceN,
    float* r, float* g, float* b, uint* a
) {
    for (uint si = 0; si < extParams->numFaceshade && si < 8u; si++) {
        __global const FaceshadeSlot* fs = &extParams->faceshades[si];
        if (fs->materialColor != 0u &&
            (fs->materialColor & 0x00FFFFFFu) != voxelRGB) continue;
        uint faceMode = ((fs->slotFlags & EXT_SLOT_LIT)  != 0u) ? 2u :
                        ((fs->slotFlags & EXT_SLOT_TINT) != 0u) ? 1u : 0u;
        apply_colorentry(fs->face[faceIdx], faceMode, r, g, b, a);
    }

    for (uint si = 0; si < extParams->numIso && si < 4u; si++) {
        __global const IsoSlot* is_ = &extParams->isos[si];
        if (is_->materialColor != 0u &&
            (is_->materialColor & 0x00FFFFFFu) != voxelRGB) continue;
        // iso role: top = Y-dominant, left = x<0, right = everything else
        float absX = fabs(faceN.x), absY = fabs(faceN.y), absZ = fabs(faceN.z);
        int isoRole = (absY > absX && absY > absZ) ? 0 :
                      (faceN.x < 0.0f)              ? 1 : 2;
        uint isoMode = ((is_->slotFlags & EXT_SLOT_LIT)  != 0u) ? 2u :
                       ((is_->slotFlags & EXT_SLOT_TINT) != 0u) ? 1u : 0u;
        apply_colorentry(is_->iso[isoRole], isoMode, r, g, b, a);
    }
}

// Linear scan for a same-volume voxel near a given model-space position
// (within 0.5 units on each axis = same unit-grid cell). O(voxelCount) per
// call, called up to 6x per voxel that needs face info (lighting, backface,
// FX, or slice/mesh raster) — O(voxelCount^2) total. This is the accepted
// cost of per-voxel occlusion-aware face exposure (see render_voxels'
// needFaceInfo block) so that lighting differentiates faces correctly
// instead of shading the whole model from one orientation-only "dominant"
// face.
inline bool voxel_exists_near(__global const Voxel* voxels, uint voxelCount, float3 target) {
    for (uint k = 0; k < voxelCount; k++) {
        float3 d = (float3)(voxels[k].x, voxels[k].y, voxels[k].z) - target;
        if (fabs(d.x) < 0.5f && fabs(d.y) < 0.5f && fabs(d.z) < 0.5f) return true;
    }
    return false;
}

// Atomic depth test using integer reinterpretation of float
// Returns true if this fragment wins the depth test (closer)
inline bool atomic_depth_test(__global volatile int* depthBuf,
                              int idx, float depth) {
    // Encode depth as integer preserving comparison order for positive values
    // IEEE 754: for positive floats, integer comparison matches float comparison
    int newDepthInt = as_int(depth);

    // CAS loop: try to write our depth if it's smaller (closer)
    int oldDepthInt = depthBuf[idx];
    while (depth < as_float(oldDepthInt)) {
        int prev = atomic_cmpxchg(depthBuf + idx, oldDepthInt, newDepthInt);
        if (prev == oldDepthInt) return true;  // we won
        oldDepthInt = prev;
    }
    return false;
}

// ---- Unit cube face vertices (model space, half-extents ±0.5) --------------
// Layout: FACE_VERTS_FLAT[face * 4 + vertex], 6 faces × 4 vertices
// Face order matches FACE_NORMALS[]: 0=front(z+), 1=back(z-), 2=right(x+),
//   3=left(x-), 4=top(y+), 5=bottom(y-)
// Winding: CCW viewed from outside (so rotated normal dots positive with camera)

__constant float3 FACE_VERTS_FLAT[24] = {
    // 0: front (z+)
    (float3)(-0.5f, -0.5f,  0.5f), (float3)( 0.5f, -0.5f,  0.5f),
    (float3)( 0.5f,  0.5f,  0.5f), (float3)(-0.5f,  0.5f,  0.5f),
    // 1: back (z-)
    (float3)( 0.5f, -0.5f, -0.5f), (float3)(-0.5f, -0.5f, -0.5f),
    (float3)(-0.5f,  0.5f, -0.5f), (float3)( 0.5f,  0.5f, -0.5f),
    // 2: right (x+)
    (float3)( 0.5f, -0.5f, -0.5f), (float3)( 0.5f, -0.5f,  0.5f),
    (float3)( 0.5f,  0.5f,  0.5f), (float3)( 0.5f,  0.5f, -0.5f),
    // 3: left (x-)
    (float3)(-0.5f, -0.5f,  0.5f), (float3)(-0.5f, -0.5f, -0.5f),
    (float3)(-0.5f,  0.5f, -0.5f), (float3)(-0.5f,  0.5f,  0.5f),
    // 4: top (y+)
    (float3)(-0.5f,  0.5f,  0.5f), (float3)( 0.5f,  0.5f,  0.5f),
    (float3)( 0.5f,  0.5f, -0.5f), (float3)(-0.5f,  0.5f, -0.5f),
    // 5: bottom (y-)
    (float3)(-0.5f, -0.5f, -0.5f), (float3)( 0.5f, -0.5f, -0.5f),
    (float3)( 0.5f, -0.5f,  0.5f), (float3)(-0.5f, -0.5f,  0.5f),
};

// ---- Scanline triangle rasterizer ------------------------------------------
// Fills triangle (sp0, sp1, sp2) in screen space with atomic depth testing.
// Uses integer scanlines; pixels at integer centers within triangle bounds.

inline void rasterize_triangle(
    float2 sp0, float2 sp1, float2 sp2, float depth,
    uint color, int fw, int fh,
    __global uint* framebuffer, __global volatile int* depthbuffer
) {
    // Sort vertices by ascending Y
    float2 tmp;
    if (sp1.y < sp0.y) { tmp = sp0; sp0 = sp1; sp1 = tmp; }
    if (sp2.y < sp0.y) { tmp = sp0; sp0 = sp2; sp2 = tmp; }
    if (sp2.y < sp1.y) { tmp = sp1; sp1 = sp2; sp2 = tmp; }

    float totalH = sp2.y - sp0.y;
    if (totalH < 0.5f) {
        // Degenerate (horizontal sliver): write centroid pixel so face isn't invisible
        int cx = (int)round((sp0.x + sp1.x + sp2.x) * 0.333333f);
        int cy = (int)round((sp0.y + sp1.y + sp2.y) * 0.333333f);
        if (cx >= 0 && cx < fw && cy >= 0 && cy < fh) {
            int idx = cy * fw + cx;
            if (atomic_depth_test(depthbuffer, idx, depth)) framebuffer[idx] = color;
        }
        return;
    }

    int y0 = max(0,    (int)ceil(sp0.y));
    int y1 = min(fh-1, (int)floor(sp2.y));

    for (int py = y0; py <= y1; py++) {
        float t_full = clamp((sp0.y == sp2.y) ? 0.0f : (float)(py - sp0.y) / totalH,
                             0.0f, 1.0f);
        float x_full = sp0.x + (sp2.x - sp0.x) * t_full;

        float x_half;
        if ((float)py <= sp1.y) {
            float seg = sp1.y - sp0.y;
            float t = (seg < 0.5f) ? 1.0f : clamp((float)(py - sp0.y) / seg, 0.0f, 1.0f);
            x_half = sp0.x + (sp1.x - sp0.x) * t;
        } else {
            float seg = sp2.y - sp1.y;
            float t = (seg < 0.5f) ? 0.0f : clamp((float)(py - sp1.y) / seg, 0.0f, 1.0f);
            x_half = sp1.x + (sp2.x - sp1.x) * t;
        }

        int px0 = max(0,    (int)floor(min(x_full, x_half) + 0.5f));
        int px1 = min(fw-1, (int)floor(max(x_full, x_half) - 0.5f + 1.0f));

        for (int px = px0; px <= px1; px++) {
            int idx = py * fw + px;
            if (atomic_depth_test(depthbuffer, idx, depth)) {
                framebuffer[idx] = color;
            }
        }
    }
}

// ---- 6 face normals (model space) -----------------------------------------

__constant float3 FACE_NORMALS[6] = {
    (float3)( 0.0f,  0.0f,  1.0f),  // 0: front  (z+)
    (float3)( 0.0f,  0.0f, -1.0f),  // 1: back   (z-)
    (float3)( 1.0f,  0.0f,  0.0f),  // 2: right  (x+)
    (float3)(-1.0f,  0.0f,  0.0f),  // 3: left   (x-)
    (float3)( 0.0f,  1.0f,  0.0f),  // 4: top    (y+)
    (float3)( 0.0f, -1.0f,  0.0f),  // 5: bottom (y-)
};

// TLR priority indices: Top(4), Left(3), Right(2), Front(0), Back(1), Bottom(5)
__constant int FACE_PRIORITY[6] = { 4, 3, 2, 0, 1, 5 };

// ---- Main kernel -----------------------------------------------------------

__kernel void render_voxels(
    __global const Voxel*           voxels,
    __global const RenderParams*    params,
    __global uint*                  framebuffer,   // RGBA packed, one uint per pixel
    __global volatile int*          depthbuffer,   // integer-reinterpreted depth
    const uint                      voxelCount,
    __global const ExtShaderParams* extParams      // FX shader params (64 B, zeros = no FX)
) {
    uint gid = get_global_id(0);
    if (gid >= voxelCount) return;

    Voxel v = voxels[gid];
    float3 pos = (float3)(v.x, v.y, v.z);
    float3 pivot = (float3)(params->middle[0], params->middle[1], params->middle[2]);

    // Rotate voxel center
    float3 rotPos = rotate_point(pos, pivot, params->rot);

    // Project to screen
    float3 screen = project(rotPos, params);
    int sx = (int)round(screen.x);
    int sy = (int)round(screen.y);
    float depth = screen.z;

    // Rasterization mode (bits 5-6 of flags)
    uint rasterMode = params->flags & FLAG_RASTER_MASK;

    // Compute voxel pixel footprint (approximate as a square)
    float pixSize;
    bool perspective = (params->flags & 1u) != 0;
    if (perspective) {
        float safeDep = fmax(0.001f, depth);
        float fovRad = params->fovDegrees * 0.0174532925f * 0.5f;
        float focalLen = ((float)params->height * 0.5f) / tan(fovRad);
        pixSize = params->voxelSize * focalLen / safeDep;
    } else {
        pixSize = params->voxelSize;
    }
    int halfPix = (int)ceil(pixSize * 0.5f);

    // Early-out for splat mode only; slice/mesh rely on per-pixel clipping
    if (rasterMode == 0u) {
        if (sx + halfPix < 0 || sx - halfPix >= (int)params->width) return;
        if (sy + halfPix < 0 || sy - halfPix >= (int)params->height) return;
    }

    // Determine dominant visible face for shading
    float3 viewDir = (float3)(0.0f, 0.0f, 1.0f); // toward camera (+Z)
    float bestDot = -2.0f;

    // Per-channel light multiplier + additive rim (defaults = flat/unlit)
    float3 colorMul = (float3)(1.0f, 1.0f, 1.0f);
    float3 rimAdd   = (float3)(0.0f, 0.0f, 0.0f);

    bool doLighting = (params->flags & FLAG_LIGHTING) != 0;
    bool doBackface = (params->flags & FLAG_BACKFACE) != 0;
    bool doDynamic  = (params->flags & FLAG_DYNAMIC) != 0;

    // Check if this is a preview voxel (ghost rendering)
    // pad[0] stores the preview voxel count as uint32 (reinterpreted)
    uint previewCount = as_uint(params->pad[0]);
    bool isPreview = (previewCount > 0 && gid >= voxelCount - previewCount);

    // Dominant face index (0-5, matching FACE_NORMALS[]) and rotated normal.
    // Computed for all non-preview voxels when lighting, backface, or FX need it.
    int  domFaceIdx = 0;
    float3 domN     = (float3)(0.0f, 0.0f, 1.0f);

    bool doFX = (extParams->numFaceshade > 0u || extParams->numIso > 0u) && !isPreview;

    // Per-face occlusion-aware exposure (camera-facing AND not covered by a
    // same-volume neighbor). Computed once here and reused by backface
    // culling, splat/slice shading+color, and FX blending, so every consumer
    // agrees on which face(s) this specific voxel actually shows — instead
    // of the whole model sharing one orientation-only "dominant" face.
    bool  exposed[6];
    float faceDot[6];
    float totalDot   = 0.0f;
    int   numExposed = 0;

    // Slice mode needs dominant face index; mesh mode needs it for per-face lighting
    bool needFaceInfo = !isPreview && (doLighting || doBackface || doFX
                        || rasterMode == FLAG_RASTER_SLICE
                        || rasterMode == FLAG_RASTER_MESH);

    if (needFaceInfo) {
        // Find the most front-facing (dominant) normal using TLR priority,
        // and in the same pass record which faces are actually exposed for
        // THIS voxel (occlusion-checked). The naive dominant normal is kept
        // as a fallback for the rare case nothing is exposed.
        for (int i = 0; i < 6; i++) {
            int fi = FACE_PRIORITY[i];
            float3 rn = rotate_dir(params->rot, FACE_NORMALS[fi]);
            float d = dot(rn, viewDir);
            if (d > bestDot) { bestDot = d; domN = rn; domFaceIdx = fi; }

            bool occluded = (d > 0.0f) && voxel_exists_near(voxels, voxelCount, pos + FACE_NORMALS[fi]);
            exposed[fi] = (d > 0.0f) && !occluded;
            faceDot[fi] = d;
            if (exposed[fi]) { totalDot += d; numExposed++; }
        }
        if (doBackface && numExposed == 0) return;

        if (numExposed > 0) {
            // Prefer this voxel's best actually-exposed face (TLR priority
            // for ties) for single-face uses: slice raster and splat shading.
            float bestExposedDot = -2.0f;
            for (int i = 0; i < 6; i++) {
                int fi = FACE_PRIORITY[i];
                if (exposed[fi] && faceDot[fi] > bestExposedDot) {
                    bestExposedDot = faceDot[fi];
                    domN = rotate_dir(params->rot, FACE_NORMALS[fi]);
                    domFaceIdx = fi;
                }
            }
        }

        if (doLighting) {
            if (doDynamic) {
                // Dynamic lighting: model-space voxel position relative to center
                float3 mrel = pos - pivot;
                shade_dynamic(domN, mrel, params, &colorMul, &rimAdd);
            } else {
                // Basic camera-facing shading — matches basic.lua exactly
                float b = shade_basic(domN, viewDir,
                                      params->lightIntensity,
                                      params->shadeIntensity);
                colorMul = (float3)(b, b, b);
            }
        }
    }

    // Preview voxels keep flat color (alpha already halved on host) via defaults.

    // Unpack base color and apply lighting
    uint rgba = v.rgba;
    float r = (float)(rgba & 0xFFu)         * colorMul.x + rimAdd.x;
    float g = (float)((rgba >> 8) & 0xFFu)  * colorMul.y + rimAdd.y;
    float b = (float)((rgba >> 16) & 0xFFu) * colorMul.z + rimAdd.z;
    uint  a = (rgba >> 24) & 0xFFu;
    if (a == 0) return; // fully transparent

    uint voxelRGB = v.rgba & 0x00FFFFFFu;  // RGB-only key for FX material matching

    if (rasterMode != FLAG_RASTER_MESH) {
        // ---- Splat / Slice: ONE final color for the whole voxel quad --------
        // Splat draws a screen-aligned square; slice draws the dominant
        // face's quad shape. Both need a single outColor, so when FX shaders
        // are active we determine THIS voxel's own genuinely exposed faces
        // (camera-facing AND not occluded by a same-volume neighbor) and
        // sRGB-blend their individually-shaded colors — matching the
        // per-voxel weighted multi-face blending Lua/Native do for rect-fill.
        if (doFX) {
            // exposed[]/faceDot[]/numExposed/totalDot were already computed
            // above (needFaceInfo block) — doFX being true guarantees that
            // block ran, since doFX is one of its trigger conditions.
            if (numExposed == 0) {
                // No unoccluded camera-facing face found (shouldn't normally
                // happen for a voxel that passed backface culling) — fall
                // back to the dominant face so it still gets a sensible color.
                exposed[domFaceIdx] = true;
                faceDot[domFaceIdx] = fmax(bestDot, 0.0001f);
                totalDot = faceDot[domFaceIdx];
                numExposed = 1;
            }

            float accR = 0.0f, accG = 0.0f, accB = 0.0f, accA = 0.0f;
            for (int f = 0; f < 6; f++) {
                if (!exposed[f]) continue;
                float fr = r, fg = g, fb = b; uint fa = a;
                float3 fn = rotate_dir(params->rot, FACE_NORMALS[f]);
                apply_fx_for_face(extParams, voxelRGB, f, fn, &fr, &fg, &fb, &fa);
                float w = faceDot[f] / totalDot;
                float lr = fr / 255.0f; lr *= lr;
                float lg = fg / 255.0f; lg *= lg;
                float lb = fb / 255.0f; lb *= lb;
                accR += lr * w; accG += lg * w; accB += lb * w;
                accA += (float)fa * w;
            }
            r = sqrt(accR) * 255.0f;
            g = sqrt(accG) * 255.0f;
            b = sqrt(accB) * 255.0f;
            a = (uint)clamp(accA + 0.5f, 0.0f, 255.0f);
        }

        uint outColor = ((uint)clamp(r, 0.0f, 255.0f))
                      | ((uint)clamp(g, 0.0f, 255.0f) << 8)
                      | ((uint)clamp(b, 0.0f, 255.0f) << 16)
                      | (a << 24);

        if (rasterMode == 0u) {
            // ---- Splat / Rect: square footprint around voxel center (default) ----
            int x0 = max(sx - halfPix, 0);
            int y0 = max(sy - halfPix, 0);
            int x1 = min(sx + halfPix, (int)params->width - 1);
            int y1 = min(sy + halfPix, (int)params->height - 1);

            for (int py = y0; py <= y1; py++) {
                for (int px = x0; px <= x1; px++) {
                    int idx = py * (int)params->width + px;
                    if (atomic_depth_test(depthbuffer, idx, depth)) {
                        framebuffer[idx] = outColor;
                    }
                }
            }
        } else {
            // ---- Slice: render only the dominant (most camera-facing) face ----
            int fw = (int)params->width;
            int fh = (int)params->height;
            int fi = domFaceIdx;

            float3 faceN = rotate_dir(params->rot, FACE_NORMALS[fi]);
            if (faceN.z >= 0.01f) {
                int base = fi * 4;
                float3 vw0 = pos + FACE_VERTS_FLAT[base + 0];
                float3 vw1 = pos + FACE_VERTS_FLAT[base + 1];
                float3 vw2 = pos + FACE_VERTS_FLAT[base + 2];
                float3 vw3 = pos + FACE_VERTS_FLAT[base + 3];

                float3 vs0 = project(rotate_point(vw0, pivot, params->rot), params);
                float3 vs1 = project(rotate_point(vw1, pivot, params->rot), params);
                float3 vs2 = project(rotate_point(vw2, pivot, params->rot), params);
                float3 vs3 = project(rotate_point(vw3, pivot, params->rot), params);

                float faceDepth = (vs0.z + vs1.z + vs2.z + vs3.z) * 0.25f;

                float2 sv0 = (float2)(vs0.x, vs0.y);
                float2 sv1 = (float2)(vs1.x, vs1.y);
                float2 sv2 = (float2)(vs2.x, vs2.y);
                float2 sv3 = (float2)(vs3.x, vs3.y);

                rasterize_triangle(sv0, sv1, sv2, faceDepth, outColor, fw, fh, framebuffer, depthbuffer);
                rasterize_triangle(sv0, sv2, sv3, faceDepth, outColor, fw, fh, framebuffer, depthbuffer);
            }
        }
    } else {
        // ---- Mesh: render all camera-facing faces as triangles, each -------
        // shaded with ITS OWN face index/normal (not the voxel's single
        // dominant face). Interior faces between adjacent solid voxels are
        // left to the depth test to resolve, same as backface-only culling
        // did before — no neighbor occlusion scan needed here.
        int fw = (int)params->width;
        int fh = (int)params->height;

        for (int fi = 0; fi < 6; fi++) {
            // Backface culling: skip faces not pointing toward camera
            float3 faceN = rotate_dir(params->rot, FACE_NORMALS[fi]);
            if (faceN.z < 0.01f) continue;

            // Shade this face with its own normal — reusing the voxel-wide
            // colorMul (computed from a single dominant face) would flatten
            // every rendered face of the model to the same brightness.
            float3 faceColorMul = (float3)(1.0f, 1.0f, 1.0f);
            float3 faceRimAdd   = (float3)(0.0f, 0.0f, 0.0f);
            if (doLighting) {
                if (doDynamic) {
                    float3 mrel = pos - pivot;
                    shade_dynamic(faceN, mrel, params, &faceColorMul, &faceRimAdd);
                } else {
                    float faceBright = shade_basic(faceN, viewDir,
                                                   params->lightIntensity,
                                                   params->shadeIntensity);
                    faceColorMul = (float3)(faceBright, faceBright, faceBright);
                }
            }
            float fr  = (float)(rgba & 0xFFu)         * faceColorMul.x + faceRimAdd.x;
            float fg  = (float)((rgba >> 8) & 0xFFu)  * faceColorMul.y + faceRimAdd.y;
            float fb  = (float)((rgba >> 16) & 0xFFu) * faceColorMul.z + faceRimAdd.z;
            uint  fa  = a;
            if (doFX) {
                apply_fx_for_face(extParams, voxelRGB, fi, faceN, &fr, &fg, &fb, &fa);
            }
            uint faceColor = ((uint)clamp(fr, 0.0f, 255.0f))
                            | ((uint)clamp(fg, 0.0f, 255.0f) << 8)
                            | ((uint)clamp(fb, 0.0f, 255.0f) << 16)
                            | (fa << 24);

            // Project the 4 face corner vertices to screen space
            int base = fi * 4;
            float3 vw0 = pos + FACE_VERTS_FLAT[base + 0];
            float3 vw1 = pos + FACE_VERTS_FLAT[base + 1];
            float3 vw2 = pos + FACE_VERTS_FLAT[base + 2];
            float3 vw3 = pos + FACE_VERTS_FLAT[base + 3];

            float3 vs0 = project(rotate_point(vw0, pivot, params->rot), params);
            float3 vs1 = project(rotate_point(vw1, pivot, params->rot), params);
            float3 vs2 = project(rotate_point(vw2, pivot, params->rot), params);
            float3 vs3 = project(rotate_point(vw3, pivot, params->rot), params);

            float faceDepth = (vs0.z + vs1.z + vs2.z + vs3.z) * 0.25f;

            float2 sv0 = (float2)(vs0.x, vs0.y);
            float2 sv1 = (float2)(vs1.x, vs1.y);
            float2 sv2 = (float2)(vs2.x, vs2.y);
            float2 sv3 = (float2)(vs3.x, vs3.y);

            // Rasterize quad as two triangles: (sv0,sv1,sv2) and (sv0,sv2,sv3)
            rasterize_triangle(sv0, sv1, sv2, faceDepth, faceColor, fw, fh, framebuffer, depthbuffer);
            rasterize_triangle(sv0, sv2, sv3, faceDepth, faceColor, fw, fh, framebuffer, depthbuffer);
        }
    }
}

// ---- Clear kernel (initialize framebuffer + depth buffer) ------------------

__kernel void clear_buffers(
    __global uint* framebuffer,
    __global int*  depthbuffer,
    const uint     pixelCount
) {
    uint gid = get_global_id(0);
    if (gid >= pixelCount) return;
    framebuffer[gid] = 0;                    // transparent black
    depthbuffer[gid] = as_int(1e30f);        // far depth (large positive float)
}

// ---- Screen-space blur post-process kernel ---------------------------------
// Blur params from ExtShaderParams:
//   flags & EXT_FLAG_BLUR_DEPTH : enable depth gating across silhouettes
//   reserved[1] : blur radius 1-8 (uint)
//   reserved[2] : blur strength 0-100 (uint)
//   reserved[3] : kernel type: 0=gaussian, 1=box, 2=triangle (uint)

__kernel void blur_framebuffer(
    __global uint*                   dst,
    __global const uint*             src,
    __global const int*              depthbuffer,
    const uint                       width,
    const uint                       height,
    __global const ExtShaderParams*  extParams
) {
    uint gid = get_global_id(0);
    uint pixelCount = width * height;
    if (gid >= pixelCount) return;

    uint srcColor = src[gid];
    // Transparent pixels: pass through without contribution
    if ((srcColor >> 24) == 0u) {
        dst[gid] = srcColor;
        return;
    }

    int px = (int)(gid % width);
    int py = (int)(gid / width);

    uint blurRadius   = extParams->blurRadius;
    uint blurStrength = extParams->blurStrength;
    uint kernelType   = extParams->blurKernelType;
    bool depthAware   = (extParams->flags & EXT_FLAG_BLUR_DEPTH) != 0u;

    if (blurRadius < 1u) blurRadius = 1u;
    if (blurRadius > 8u) blurRadius = 8u;

    float strength = (float)blurStrength / 100.0f;
    int   radius   = (int)blurRadius;

    float accR = 0.0f, accG = 0.0f, accB = 0.0f, totalWeight = 0.0f;

    float centerDepth = as_float(depthbuffer[gid]);

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = px + dx;
            int ny = py + dy;
            if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) continue;

            int  nidx   = ny * (int)width + nx;
            uint ncolor = src[nidx];
            if ((ncolor >> 24) == 0u) continue;  // skip transparent neighbors

            float dist = sqrt((float)(dx*dx + dy*dy));
            if (dist > (float)radius + 0.5f) continue;  // circle crop

            float weight;
            if (kernelType == 1u) {
                weight = 1.0f;  // box
            } else if (kernelType == 2u) {
                weight = 1.0f - dist / (float)(radius + 1);  // triangle
                if (weight < 0.0f) weight = 0.0f;
            } else {
                float sigma = (float)radius * 0.5f;
                if (sigma < 0.5f) sigma = 0.5f;
                weight = exp(-(dist * dist) / (2.0f * sigma * sigma));  // gaussian
            }

            if (depthAware) {
                float ndepth    = as_float(depthbuffer[nidx]);
                float depthDiff = fabs(centerDepth - ndepth);
                // Gate neighbors more than 2 depth units away
                if (depthDiff >= 2.0f) {
                    weight = 0.0f;
                } else {
                    weight *= 1.0f - depthDiff * 0.5f;
                }
            }

            if (weight <= 0.0f) continue;

            accR += (float)(ncolor & 0xFFu)         * weight;
            accG += (float)((ncolor >> 8)  & 0xFFu) * weight;
            accB += (float)((ncolor >> 16) & 0xFFu) * weight;
            totalWeight += weight;
        }
    }

    uint outColor = srcColor;
    if (totalWeight > 0.0f) {
        float blurR = accR / totalWeight;
        float blurG = accG / totalWeight;
        float blurB = accB / totalWeight;

        float origR = (float)(srcColor & 0xFFu);
        float origG = (float)((srcColor >> 8)  & 0xFFu);
        float origB = (float)((srcColor >> 16) & 0xFFu);
        uint  origA = (srcColor >> 24) & 0xFFu;

        outColor = ((uint)clamp(origR + (blurR - origR) * strength, 0.0f, 255.0f))
                 | ((uint)clamp(origG + (blurG - origG) * strength, 0.0f, 255.0f) << 8)
                 | ((uint)clamp(origB + (blurB - origB) * strength, 0.0f, 255.0f) << 16)
                 | (origA << 24);
    }

    dst[gid] = outColor;
}

)CL";

#endif // ASEVOXEL_KERNEL_H
