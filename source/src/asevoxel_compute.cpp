// ============================================================================
// AseVoxel Compute — Headless OpenCL voxel renderer
// ============================================================================
// Usage:
//   asevoxel_compute [--probe] [--stdin-file <path>] [--kernel-file <path>]
//
// Protocol:
//   Input:  RenderParams (128 bytes) + Voxel[N] (16 bytes each)  via stdin or file
//   Output: uint32 width + uint32 height + RGBA8[W×H]            to stdout
//   Diag:   JSON                                                  to stderr
//
// Exit codes:
//   0 = success
//   1 = generic error
//   2 = no OpenCL platform
//   3 = no usable device
//   4 = kernel compile error
//   5 = I/O error
// ============================================================================

#include "platform.h"
#include "opencl_init.h"
#include "io_pipeline.h"
#include "render_gpu.h"
#include "daemon.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// ============================================================================
// Main
// ============================================================================

static constexpr uint16_t DEFAULT_DAEMON_PORT = 9010;

int main(int argc, char* argv[]) {
    std::string stdinFile;
    bool probeMode = false;
    bool daemonMode = false;
    uint16_t daemonPort = DEFAULT_DAEMON_PORT;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--probe") == 0) {
            probeMode = true;
        } else if (strcmp(argv[i], "--stdin-file") == 0 && i + 1 < argc) {
            stdinFile = argv[++i];
        } else if (strcmp(argv[i], "--daemon") == 0) {
            daemonMode = true;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            daemonPort = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr,
                "AseVoxel Compute — Headless OpenCL voxel renderer\n"
                "Usage: %s [--probe] [--stdin-file <path>] [--daemon] [--port <N>]\n"
                "  --probe         Print OpenCL device info (JSON to stderr) and exit\n"
                "  --stdin-file F  Read input from file F instead of stdin\n"
                "  --daemon        Run as persistent WebSocket render server\n"
                "  --port N        Daemon port (default: %u)\n"
                "  --help          Show this help\n",
                argv[0], DEFAULT_DAEMON_PORT);
            return 0;
        } else {
            fprintf(stderr, "{\"error\":\"unknown_arg\",\"arg\":\"%s\"}\n", argv[i]);
            return EXIT_GENERIC;
        }
    }

    // Probe mode: just print device info and exit
    if (probeMode) {
        return opencl_probe();
    }

    // Initialize OpenCL
    auto t0 = std::chrono::steady_clock::now();
    ComputeContext ctx;
    int initResult = opencl_init(ctx);
    if (initResult != 0) return initResult;

    auto t1 = std::chrono::steady_clock::now();
    double initMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Daemon mode: persistent WebSocket render server
    if (daemonMode) {
        fprintf(stderr, "{\"init_ms\":%.1f}\n", initMs);
        return run_daemon(daemonPort, ctx);
    }

    // One-shot mode: read input, render, write output
    RenderInput input;
    int ioResult;
    if (!stdinFile.empty()) {
        ioResult = read_input(stdinFile, input);
    } else {
        ioResult = read_input_stdin(input);
    }
    if (ioResult != 0) return ioResult;

    auto t2 = std::chrono::steady_clock::now();
    double readMs = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Render
    std::vector<uint8_t> pixels;
    int renderResult = render_gpu(ctx, input, pixels);
    if (renderResult != 0) return renderResult;

    auto t3 = std::chrono::steady_clock::now();
    double renderMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Write output
    int writeResult = write_output(input.params.width, input.params.height, pixels.data());
    if (writeResult != 0) return writeResult;

    auto t4 = std::chrono::steady_clock::now();
    double writeMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double totalMs = std::chrono::duration<double, std::milli>(t4 - t0).count();

    // Diagnostics to stderr
    fprintf(stderr, "{\"ok\":true,"
            "\"device\":\"%s\","
            "\"voxels\":%u,"
            "\"resolution\":\"%ux%u\","
            "\"init_ms\":%.1f,"
            "\"read_ms\":%.1f,"
            "\"render_ms\":%.1f,"
            "\"write_ms\":%.1f,"
            "\"total_ms\":%.1f}\n",
            ctx.deviceName.c_str(),
            input.params.voxelCount,
            input.params.width, input.params.height,
            initMs, readMs, renderMs, writeMs, totalMs);

    return 0;
}
