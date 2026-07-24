#ifndef ASEVOXEL_OPENCL_INIT_H
#define ASEVOXEL_OPENCL_INIT_H

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#include <string>
#include <cstdint>

// RAII wrapper for OpenCL context + queue + device
struct ComputeContext {
    cl_platform_id platform  = nullptr;
    cl_device_id   device    = nullptr;
    cl_context     context   = nullptr;
    cl_command_queue queue   = nullptr;
    cl_program     program   = nullptr;
    cl_kernel      renderKernel = nullptr;
    cl_kernel      clearKernel  = nullptr;
    cl_kernel      blurKernel   = nullptr;

    std::string platformName;
    std::string deviceName;
    uint64_t    globalMemBytes = 0;
    uint32_t    computeUnits   = 0;
    uint32_t    maxWorkGroupSize = 0;

    bool valid() const { return context != nullptr && queue != nullptr; }

    // Release all resources
    void release();
    ~ComputeContext() { release(); }

    // Non-copyable
    ComputeContext() = default;
    ComputeContext(const ComputeContext&) = delete;
    ComputeContext& operator=(const ComputeContext&) = delete;
    ComputeContext(ComputeContext&& o) noexcept;
    ComputeContext& operator=(ComputeContext&& o) noexcept;
};

// Initialize OpenCL: find best device, create context + queue, build kernels.
// Returns exit code (0 = success, EXIT_NO_OPENCL/EXIT_NO_DEVICE/EXIT_KERNEL_ERROR on failure).
// On success, writes to `ctx`. On failure, writes JSON diagnostic to stderr.
int opencl_init(ComputeContext& ctx);

// Print JSON probe info to stderr and return 0. Used by --probe flag.
int opencl_probe();

#endif // ASEVOXEL_OPENCL_INIT_H
