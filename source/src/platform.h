#ifndef ASEVOXEL_PLATFORM_H
#define ASEVOXEL_PLATFORM_H

#include <cstdint>

// ============================================================================
// Cross-platform macros for AseVoxel Compute
// ============================================================================

#ifdef _WIN32
  #include <io.h>
  #include <fcntl.h>
  #define SET_BINARY_MODE_STDIN()  _setmode(_fileno(stdin), _O_BINARY)
  #define SET_BINARY_MODE_STDOUT() _setmode(_fileno(stdout), _O_BINARY)
  #define PATH_SEP "\\"
#else
  #define SET_BINARY_MODE_STDIN()  ((void)0)
  #define SET_BINARY_MODE_STDOUT() ((void)0)
  #define PATH_SEP "/"
#endif

// Exit codes (Lua caller checks these for fallback decisions)
constexpr int EXIT_OK           = 0;
constexpr int EXIT_GENERIC      = 1;
constexpr int EXIT_NO_OPENCL    = 2;
constexpr int EXIT_NO_DEVICE    = 3;
constexpr int EXIT_KERNEL_ERROR = 4;
constexpr int EXIT_IO_ERROR     = 5;

// Hard limits
constexpr uint32_t MAX_VOXELS       = 134217728; // 512^3
constexpr uint32_t MAX_FRAMEBUFFER  = 8192 * 8192;

#endif // ASEVOXEL_PLATFORM_H
