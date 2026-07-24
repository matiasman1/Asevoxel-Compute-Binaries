#ifndef ASEVOXEL_RENDER_GPU_H
#define ASEVOXEL_RENDER_GPU_H

#include "opencl_init.h"
#include "io_pipeline.h"
#include <vector>
#include <cstdint>

// Render voxels on GPU. Returns 0 on success, EXIT_GENERIC on failure.
// outputPixels is resized to width*height*4 bytes (RGBA8).
int render_gpu(ComputeContext& ctx, const RenderInput& input, std::vector<uint8_t>& outputPixels);

#endif // ASEVOXEL_RENDER_GPU_H
