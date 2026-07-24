#ifndef ASEVOXEL_DAEMON_H
#define ASEVOXEL_DAEMON_H

#include "opencl_init.h"
#include <cstdint>

// Run the WebSocket daemon on localhost:port.
// Blocks until shutdown command or fatal error.
// Returns exit code (0 = clean shutdown).
int run_daemon(uint16_t port, ComputeContext& ctx);

#endif // ASEVOXEL_DAEMON_H
