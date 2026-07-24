#include "opencl_init.h"
#include "platform.h"
#include "kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/stat.h>

#ifdef _WIN32
  #include <direct.h>
  #include <shlobj.h>
  #define MKDIR(p) _mkdir(p)
#else
  #include <unistd.h>
  #define MKDIR(p) mkdir(p, 0755)
#endif

// ============================================================================
// Helpers
// ============================================================================

static std::string get_string(cl_device_id dev, cl_device_info param) {
    size_t len = 0;
    clGetDeviceInfo(dev, param, 0, nullptr, &len);
    std::string s(len, '\0');
    clGetDeviceInfo(dev, param, len, &s[0], nullptr);
    // trim trailing null
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static std::string get_platform_string(cl_platform_id plat, cl_platform_info param) {
    size_t len = 0;
    clGetPlatformInfo(plat, param, 0, nullptr, &len);
    std::string s(len, '\0');
    clGetPlatformInfo(plat, param, len, &s[0], nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static int score_device(cl_device_id dev) {
    cl_device_type type;
    clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);

    int base = 0;
    if (type & CL_DEVICE_TYPE_GPU)         base = 100;
    else if (type & CL_DEVICE_TYPE_ACCELERATOR) base = 50;
    else if (type & CL_DEVICE_TYPE_CPU)    base = 10;

    cl_uint units = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr);

    cl_ulong mem = 0;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);

    return base + (int)units + (int)(mem >> 30); // +1 per compute unit, +1 per GB
}

// ============================================================================
// Kernel cache
// ============================================================================

static std::string cache_dir() {
    std::string dir;
#ifdef _WIN32
    char* appdata = getenv("LOCALAPPDATA");
    if (appdata) {
        dir = std::string(appdata) + "\\asevoxel";
    }
#else
    char* home = getenv("HOME");
    if (home) {
        dir = std::string(home) + "/.cache/asevoxel";
    }
#endif
    return dir;
}

static uint32_t simple_hash(const char* str) {
    uint32_t h = 5381;
    while (*str) {
        h = ((h << 5) + h) + (unsigned char)*str;
        str++;
    }
    return h;
}

static std::string kernel_cache_path(cl_device_id dev) {
    std::string dir = cache_dir();
    if (dir.empty()) return "";
    std::string devName = get_string(dev, CL_DEVICE_NAME);
    std::string devVer = get_string(dev, CL_DEVICE_VERSION);
    std::string key = devName + "|" + devVer + "|" + KERNEL_SOURCE;
    uint32_t h = simple_hash(key.c_str());
    char fname[64];
    snprintf(fname, sizeof(fname), "kernel_%08x.bin", h);
    return dir + PATH_SEP + fname;
}

static std::vector<unsigned char> read_binary_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return {}; }
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(sz);
    size_t rd = fread(buf.data(), 1, sz, f);
    fclose(f);
    if ((long)rd != sz) return {};
    return buf;
}

static void write_binary_file(const std::string& path, const void* data, size_t len) {
    std::string dir = cache_dir();
    if (!dir.empty()) MKDIR(dir.c_str()); // ensure dir exists
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(data, 1, len, f);
    fclose(f);
}

// ============================================================================
// Build kernels (with cache)
// ============================================================================

static int build_kernels(ComputeContext& ctx) {
    cl_int err;

    // Try loading from cache first
    std::string cachePath = kernel_cache_path(ctx.device);
    if (!cachePath.empty()) {
        auto cached = read_binary_file(cachePath);
        if (!cached.empty()) {
            size_t binSize = cached.size();
            const unsigned char* binData = cached.data();
            cl_int binStatus;
            ctx.program = clCreateProgramWithBinary(
                ctx.context, 1, &ctx.device,
                &binSize, &binData, &binStatus, &err
            );
            if (err == CL_SUCCESS && binStatus == CL_SUCCESS) {
                err = clBuildProgram(ctx.program, 1, &ctx.device, nullptr, nullptr, nullptr);
                if (err == CL_SUCCESS) {
                    ctx.renderKernel = clCreateKernel(ctx.program, "render_voxels", &err);
                    if (err == CL_SUCCESS) {
                        ctx.clearKernel = clCreateKernel(ctx.program, "clear_buffers", &err);
                        if (err == CL_SUCCESS) {
                            ctx.blurKernel = clCreateKernel(ctx.program, "blur_framebuffer", &err);
                            if (err == CL_SUCCESS) return 0;
                        }
                    }
                }
                // If cached binary failed for any reason, fall through to source build
                clReleaseProgram(ctx.program);
                ctx.program      = nullptr;
                ctx.renderKernel = nullptr;
                ctx.clearKernel  = nullptr;
                ctx.blurKernel   = nullptr;
            }
        }
    }

    // Build from source
    const char* src = KERNEL_SOURCE;
    size_t srcLen = strlen(KERNEL_SOURCE);
    ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"create_program_failed\",\"cl_error\":%d}\n", err);
        return EXIT_KERNEL_ERROR;
    }

    err = clBuildProgram(ctx.program, 1, &ctx.device, "-cl-std=CL1.2 -cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize = 0;
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string buildLog(logSize, '\0');
        clGetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);
        fprintf(stderr, "{\"error\":\"kernel_build_failed\",\"cl_error\":%d,\"log\":\"%s\"}\n",
                err, buildLog.c_str());
        return EXIT_KERNEL_ERROR;
    }

    ctx.renderKernel = clCreateKernel(ctx.program, "render_voxels", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"create_render_kernel_failed\",\"cl_error\":%d}\n", err);
        return EXIT_KERNEL_ERROR;
    }

    ctx.clearKernel = clCreateKernel(ctx.program, "clear_buffers", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"create_clear_kernel_failed\",\"cl_error\":%d}\n", err);
        return EXIT_KERNEL_ERROR;
    }

    ctx.blurKernel = clCreateKernel(ctx.program, "blur_framebuffer", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"create_blur_kernel_failed\",\"cl_error\":%d}\n", err);
        return EXIT_KERNEL_ERROR;
    }

    // Cache the compiled binary
    if (!cachePath.empty()) {
        size_t binSize = 0;
        clGetProgramInfo(ctx.program, CL_PROGRAM_BINARY_SIZES, sizeof(binSize), &binSize, nullptr);
        if (binSize > 0) {
            std::vector<unsigned char> binary(binSize);
            unsigned char* binPtr = binary.data();
            clGetProgramInfo(ctx.program, CL_PROGRAM_BINARIES, sizeof(binPtr), &binPtr, nullptr);
            write_binary_file(cachePath, binary.data(), binSize);
        }
    }

    return 0;
}

// ============================================================================
// ComputeContext implementation
// ============================================================================

void ComputeContext::release() {
    if (blurKernel)   { clReleaseKernel(blurKernel);   blurKernel = nullptr; }
    if (clearKernel)  { clReleaseKernel(clearKernel);  clearKernel = nullptr; }
    if (renderKernel) { clReleaseKernel(renderKernel); renderKernel = nullptr; }
    if (program)      { clReleaseProgram(program);     program = nullptr; }
    if (queue)        { clReleaseCommandQueue(queue);   queue = nullptr; }
    if (context)      { clReleaseContext(context);      context = nullptr; }
}

ComputeContext::ComputeContext(ComputeContext&& o) noexcept
    : platform(o.platform), device(o.device), context(o.context),
      queue(o.queue), program(o.program),
      renderKernel(o.renderKernel), clearKernel(o.clearKernel),
      blurKernel(o.blurKernel),
      platformName(std::move(o.platformName)),
      deviceName(std::move(o.deviceName)),
      globalMemBytes(o.globalMemBytes),
      computeUnits(o.computeUnits),
      maxWorkGroupSize(o.maxWorkGroupSize)
{
    o.context = nullptr; o.queue = nullptr; o.program = nullptr;
    o.renderKernel = nullptr; o.clearKernel = nullptr; o.blurKernel = nullptr;
}

ComputeContext& ComputeContext::operator=(ComputeContext&& o) noexcept {
    if (this != &o) {
        release();
        platform = o.platform; device = o.device;
        context = o.context; queue = o.queue; program = o.program;
        renderKernel = o.renderKernel; clearKernel = o.clearKernel;
        blurKernel = o.blurKernel;
        platformName = std::move(o.platformName);
        deviceName = std::move(o.deviceName);
        globalMemBytes = o.globalMemBytes;
        computeUnits = o.computeUnits;
        maxWorkGroupSize = o.maxWorkGroupSize;
        o.context = nullptr; o.queue = nullptr; o.program = nullptr;
        o.renderKernel = nullptr; o.clearKernel = nullptr; o.blurKernel = nullptr;
    }
    return *this;
}

// ============================================================================
// Public API
// ============================================================================

int opencl_init(ComputeContext& ctx) {
    cl_int err;

    // Enumerate platforms
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        fprintf(stderr, "{\"error\":\"no_opencl\",\"platforms\":0}\n");
        return EXIT_NO_OPENCL;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Enumerate all devices across all platforms, score them
    struct DeviceCandidate {
        cl_platform_id plat;
        cl_device_id   dev;
        int            score;
    };
    std::vector<DeviceCandidate> candidates;

    for (auto& plat : platforms) {
        cl_uint numDevices = 0;
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        if (numDevices == 0) continue;

        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);

        for (auto& dev : devices) {
            candidates.push_back({plat, dev, score_device(dev)});
        }
    }

    if (candidates.empty()) {
        fprintf(stderr, "{\"error\":\"no_device\",\"platforms\":%u,\"devices\":0}\n", numPlatforms);
        return EXIT_NO_DEVICE;
    }

    // Pick highest-scoring device
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    auto& best = candidates[0];

    ctx.platform = best.plat;
    ctx.device = best.dev;
    ctx.platformName = get_platform_string(ctx.platform, CL_PLATFORM_NAME);
    ctx.deviceName = get_string(ctx.device, CL_DEVICE_NAME);

    cl_ulong mem = 0;
    clGetDeviceInfo(ctx.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    ctx.globalMemBytes = mem;

    cl_uint units = 0;
    clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr);
    ctx.computeUnits = units;

    size_t wgs = 0;
    clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr);
    ctx.maxWorkGroupSize = (uint32_t)wgs;

    // Create context
    ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"context_failed\",\"cl_error\":%d}\n", err);
        return EXIT_NO_DEVICE;
    }

    // Create command queue
    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"queue_failed\",\"cl_error\":%d}\n", err);
        return EXIT_NO_DEVICE;
    }

    // Build kernels
    int kErr = build_kernels(ctx);
    if (kErr != 0) return kErr;

    return 0;
}

int opencl_probe() {
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) numPlatforms = 0;

    fprintf(stderr, "{\"probe\":true,\"platforms\":[");

    if (numPlatforms > 0) {
        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

        for (cl_uint pi = 0; pi < numPlatforms; pi++) {
            if (pi > 0) fprintf(stderr, ",");
            std::string pName = get_platform_string(platforms[pi], CL_PLATFORM_NAME);
            std::string pVer = get_platform_string(platforms[pi], CL_PLATFORM_VERSION);
            fprintf(stderr, "{\"name\":\"%s\",\"version\":\"%s\",\"devices\":[",
                    pName.c_str(), pVer.c_str());

            cl_uint numDevices = 0;
            clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);

            if (numDevices > 0) {
                std::vector<cl_device_id> devices(numDevices);
                clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);

                for (cl_uint di = 0; di < numDevices; di++) {
                    if (di > 0) fprintf(stderr, ",");
                    std::string dName = get_string(devices[di], CL_DEVICE_NAME);
                    std::string dVer = get_string(devices[di], CL_DEVICE_VERSION);

                    cl_device_type dtype;
                    clGetDeviceInfo(devices[di], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
                    const char* typeStr = "unknown";
                    if (dtype & CL_DEVICE_TYPE_GPU)         typeStr = "gpu";
                    else if (dtype & CL_DEVICE_TYPE_CPU)    typeStr = "cpu";
                    else if (dtype & CL_DEVICE_TYPE_ACCELERATOR) typeStr = "accelerator";

                    cl_uint units = 0;
                    clGetDeviceInfo(devices[di], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr);
                    cl_ulong mem = 0;
                    clGetDeviceInfo(devices[di], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);

                    fprintf(stderr, "{\"name\":\"%s\",\"version\":\"%s\",\"type\":\"%s\","
                            "\"compute_units\":%u,\"global_mem_mb\":%llu,\"score\":%d}",
                            dName.c_str(), dVer.c_str(), typeStr,
                            units, (unsigned long long)(mem >> 20),
                            score_device(devices[di]));
                }
            }
            fprintf(stderr, "]}");
        }
    }

    fprintf(stderr, "]}\n");
    return 0;
}
