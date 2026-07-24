#include "io_pipeline.h"
#include "platform.h"

#include <cstdio>
#include <cstring>

// ============================================================================
// Read helpers
// ============================================================================

static bool read_exact(FILE* f, void* buf, size_t count) {
    size_t total = 0;
    auto* p = static_cast<uint8_t*>(buf);
    while (total < count) {
        size_t rd = fread(p + total, 1, count - total, f);
        if (rd == 0) return false; // EOF or error
        total += rd;
    }
    return true;
}

static int read_from_file(FILE* f, RenderInput& input) {
    // Read fixed-size header
    if (!read_exact(f, &input.params, sizeof(RenderParams))) {
        fprintf(stderr, "{\"error\":\"read_params_failed\",\"expected\":%zu}\n",
                sizeof(RenderParams));
        return EXIT_IO_ERROR;
    }

    // Validate
    if (input.params.voxelCount > MAX_VOXELS) {
        fprintf(stderr, "{\"error\":\"too_many_voxels\",\"count\":%u,\"max\":%u}\n",
                input.params.voxelCount, MAX_VOXELS);
        return EXIT_IO_ERROR;
    }
    if (input.params.width == 0 || input.params.height == 0) {
        fprintf(stderr, "{\"error\":\"invalid_dimensions\",\"width\":%u,\"height\":%u}\n",
                input.params.width, input.params.height);
        return EXIT_IO_ERROR;
    }
    if ((uint64_t)input.params.width * input.params.height > MAX_FRAMEBUFFER) {
        fprintf(stderr, "{\"error\":\"framebuffer_too_large\",\"pixels\":%llu,\"max\":%u}\n",
                (unsigned long long)input.params.width * input.params.height, MAX_FRAMEBUFFER);
        return EXIT_IO_ERROR;
    }

    // Read voxel data
    if (input.params.voxelCount > 0) {
        input.voxels.resize(input.params.voxelCount);
        size_t voxelBytes = (size_t)input.params.voxelCount * sizeof(Voxel);
        if (!read_exact(f, input.voxels.data(), voxelBytes)) {
            fprintf(stderr, "{\"error\":\"read_voxels_failed\",\"expected\":%zu}\n", voxelBytes);
            return EXIT_IO_ERROR;
        }
    }

    // Try to read optional ExtShaderParams (64 bytes). Absence = no FX shaders active.
    memset(&input.extShaderParams, 0, sizeof(ExtShaderParams));
    read_exact(f, &input.extShaderParams, sizeof(ExtShaderParams));
    // Intentionally ignoring read failure — backward-compatible with old streams.

    return 0;
}

// ============================================================================
// Public API
// ============================================================================

int read_input(const std::string& path, RenderInput& input) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "{\"error\":\"open_failed\",\"path\":\"%s\"}\n", path.c_str());
        return EXIT_IO_ERROR;
    }
    int result = read_from_file(f, input);
    fclose(f);
    return result;
}

int read_input_stdin(RenderInput& input) {
    SET_BINARY_MODE_STDIN();
    return read_from_file(stdin, input);
}

int write_output(uint32_t width, uint32_t height, const uint8_t* pixels) {
    SET_BINARY_MODE_STDOUT();

    // Write header: width, height (8 bytes)
    if (fwrite(&width, sizeof(uint32_t), 1, stdout) != 1) {
        fprintf(stderr, "{\"error\":\"write_width_failed\"}\n");
        return EXIT_IO_ERROR;
    }
    if (fwrite(&height, sizeof(uint32_t), 1, stdout) != 1) {
        fprintf(stderr, "{\"error\":\"write_height_failed\"}\n");
        return EXIT_IO_ERROR;
    }

    // Write pixel data
    size_t pixelBytes = (size_t)width * height * 4;
    if (pixelBytes > 0) {
        size_t written = fwrite(pixels, 1, pixelBytes, stdout);
        if (written != pixelBytes) {
            fprintf(stderr, "{\"error\":\"write_pixels_failed\",\"written\":%zu,\"expected\":%zu}\n",
                    written, pixelBytes);
            return EXIT_IO_ERROR;
        }
    }

    fflush(stdout);
    return 0;
}
