// ============================================================================
// AseVoxel Daemon — persistent WebSocket render server
// ============================================================================
// Protocol over WebSocket binary frames:
//
// REQUEST (client → server):
//   Byte 0:    command
//     0x01 = render: followed by RenderParams(128) + Voxels(N×16)
//     0x02 = probe:  no payload
//     0xFF = shutdown: no payload
//
// RESPONSE (server → client):
//   Byte 0:    status
//     0x00 = render ok: [4B width][4B height][W×H×4 RGBA]
//     0x01 = error:     [1B code][rest: UTF-8 message]
//     0x02 = probe ok:  [rest: JSON text]
// ============================================================================

#include "daemon.h"
#include "ws_server.h"
#include "render_gpu.h"
#include "io_pipeline.h"
#include "platform.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

// Protocol constants
static constexpr uint8_t CMD_RENDER   = 0x01;
static constexpr uint8_t CMD_PROBE    = 0x02;
static constexpr uint8_t CMD_SHUTDOWN = 0xFF;

static constexpr uint8_t STATUS_OK    = 0x00;
static constexpr uint8_t STATUS_ERROR = 0x01;
static constexpr uint8_t STATUS_PROBE = 0x02;

// Signal handling for graceful shutdown
#ifdef _WIN32
  #include <windows.h>
  static volatile LONG g_daemon_running = 1;
  static BOOL WINAPI daemon_console_handler(DWORD ev) {
      if (ev == CTRL_C_EVENT || ev == CTRL_CLOSE_EVENT) {
          InterlockedExchange(&g_daemon_running, 0);
          return TRUE;
      }
      return FALSE;
  }
  static void install_signal_handlers() { SetConsoleCtrlHandler(daemon_console_handler, TRUE); }
  static bool daemon_should_run() { return InterlockedCompareExchange(&g_daemon_running, 1, 1) == 1; }
#else
  #include <signal.h>
  static volatile sig_atomic_t g_daemon_running = 1;
  static void daemon_signal_handler(int) { g_daemon_running = 0; }
  static void install_signal_handlers() {
      signal(SIGINT,  daemon_signal_handler);
      signal(SIGTERM, daemon_signal_handler);
  }
  static bool daemon_should_run() { return g_daemon_running != 0; }
#endif

// ============================================================================
// Send helpers
// ============================================================================

static bool send_error(WsServer& ws, uint8_t code, const char* msg) {
    size_t msgLen = strlen(msg);
    std::vector<uint8_t> resp(2 + msgLen);
    resp[0] = STATUS_ERROR;
    resp[1] = code;
    memcpy(resp.data() + 2, msg, msgLen);
    return ws.sendBinary(resp.data(), resp.size());
}

static bool send_render_result(WsServer& ws, uint32_t w, uint32_t h, const std::vector<uint8_t>& pixels) {
    size_t respLen = 1 + 4 + 4 + pixels.size();
    std::vector<uint8_t> resp(respLen);
    resp[0] = STATUS_OK;
    memcpy(resp.data() + 1, &w, 4);
    memcpy(resp.data() + 5, &h, 4);
    if (!pixels.empty())
        memcpy(resp.data() + 9, pixels.data(), pixels.size());
    return ws.sendBinary(resp.data(), resp.size());
}

static bool send_probe(WsServer& ws, const ComputeContext& ctx) {
    char buf[1024];
    int n = snprintf(buf, sizeof(buf),
        "{\"daemon\":true,\"device\":\"%s\",\"compute_units\":%u,"
        "\"global_mem_mb\":%u,\"max_work_group\":%u}",
        ctx.deviceName.c_str(), ctx.computeUnits,
        (uint32_t)(ctx.globalMemBytes / (1024*1024)),
        ctx.maxWorkGroupSize);
    std::vector<uint8_t> resp(1 + n);
    resp[0] = STATUS_PROBE;
    memcpy(resp.data() + 1, buf, n);
    return ws.sendBinary(resp.data(), resp.size());
}

// ============================================================================
// Handle a render request
// ============================================================================

static bool handle_render(WsServer& ws, ComputeContext& ctx,
                          const uint8_t* data, size_t len) {
    // Need at least 1 (command) + 128 (params) = 129 bytes
    if (len < 129) {
        send_error(ws, EXIT_IO_ERROR, "render request too short");
        return true; // connection still alive
    }

    auto t0 = std::chrono::steady_clock::now();

    // Parse RenderParams from bytes 1..128
    RenderInput input;
    memcpy(&input.params, data + 1, sizeof(RenderParams));

    // Validate
    if (input.params.voxelCount > MAX_VOXELS) {
        send_error(ws, EXIT_IO_ERROR, "too many voxels");
        return true;
    }
    if (input.params.width == 0 || input.params.height == 0) {
        send_error(ws, EXIT_IO_ERROR, "invalid dimensions");
        return true;
    }
    if ((uint64_t)input.params.width * input.params.height > MAX_FRAMEBUFFER) {
        send_error(ws, EXIT_IO_ERROR, "framebuffer too large");
        return true;
    }

    // Parse voxels from bytes 129+
    size_t expectedVoxelBytes = (size_t)input.params.voxelCount * sizeof(Voxel);
    if (len < 129 + expectedVoxelBytes) {
        send_error(ws, EXIT_IO_ERROR, "incomplete voxel data");
        return true;
    }

    if (input.params.voxelCount > 0) {
        input.voxels.resize(input.params.voxelCount);
        memcpy(input.voxels.data(), data + 129, expectedVoxelBytes);
    }

    // Try to parse optional ExtShaderParams (64 bytes after voxels). Absence = all-zeros.
    memset(&input.extShaderParams, 0, sizeof(ExtShaderParams));
    size_t extOffset = 129 + expectedVoxelBytes;
    if (len >= extOffset + sizeof(ExtShaderParams)) {
        memcpy(&input.extShaderParams, data + extOffset, sizeof(ExtShaderParams));
    }

    // Render
    std::vector<uint8_t> pixels;
    int renderResult = render_gpu(ctx, input, pixels);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (renderResult != 0) {
        send_error(ws, (uint8_t)renderResult, "gpu render failed");
        return true;
    }

    // Send result
    bool sendOk = send_render_result(ws, input.params.width, input.params.height, pixels);

    // Diagnostics to stderr
    fprintf(stderr, "{\"ok\":true,\"voxels\":%u,\"resolution\":\"%ux%u\","
            "\"render_ms\":%.1f}\n",
            input.params.voxelCount,
            input.params.width, input.params.height, ms);

    return sendOk;
}

// ============================================================================
// Main daemon loop
// ============================================================================

int run_daemon(uint16_t port, ComputeContext& ctx) {
    install_signal_handlers();

    WsServer server;
    if (!server.start(port)) {
        fprintf(stderr, "{\"error\":\"bind_failed\",\"port\":%u}\n", port);
        return EXIT_GENERIC;
    }

    fprintf(stderr, "{\"daemon\":true,\"port\":%u,\"device\":\"%s\","
            "\"status\":\"listening\"}\n",
            server.port(), ctx.deviceName.c_str());
    fflush(stderr);

    uint32_t totalRenders = 0;

    while (daemon_should_run()) {
        fprintf(stderr, "{\"status\":\"waiting_for_client\"}\n");
        fflush(stderr);

        if (!server.acceptClient()) {
            if (!daemon_should_run()) break;
            fprintf(stderr, "{\"error\":\"accept_failed\"}\n");
            continue;
        }

        fprintf(stderr, "{\"status\":\"client_connected\"}\n");
        fflush(stderr);

        // Client session loop
        while (daemon_should_run() && server.hasClient()) {
            WsMessage msg;
            if (!server.readMessage(msg)) {
                // Client disconnected
                fprintf(stderr, "{\"status\":\"client_disconnected\","
                        "\"total_renders\":%u}\n", totalRenders);
                fflush(stderr);
                server.closeClient();
                break;
            }

            switch (msg.opcode) {
                case WsMessage::CLOSE:
                    server.closeClient();
                    break;

                case WsMessage::PING:
                    server.sendPong(msg.data.data(), msg.data.size());
                    break;

                case WsMessage::BINARY: {
                    if (msg.data.empty()) break;
                    uint8_t command = msg.data[0];

                    if (command == CMD_RENDER) {
                        handle_render(server, ctx, msg.data.data(), msg.data.size());
                        totalRenders++;
                    } else if (command == CMD_PROBE) {
                        send_probe(server, ctx);
                    } else if (command == CMD_SHUTDOWN) {
                        fprintf(stderr, "{\"status\":\"shutdown_requested\","
                                "\"total_renders\":%u}\n", totalRenders);
                        fflush(stderr);
                        server.closeClient();
                        server.stop();
                        return EXIT_OK;
                    }
                    break;
                }

                default:
                    break;
            }
        }
    }

    fprintf(stderr, "{\"status\":\"daemon_stopped\",\"total_renders\":%u}\n", totalRenders);
    fflush(stderr);
    server.stop();
    return EXIT_OK;
}
