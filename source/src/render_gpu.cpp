// ============================================================================
// GPU render pipeline — shared by one-shot mode and daemon mode
// ============================================================================

#include "render_gpu.h"
#include "platform.h"

#include <cstdio>

int render_gpu(ComputeContext& ctx, const RenderInput& input, std::vector<uint8_t>& outputPixels) {
    cl_int err = CL_SUCCESS;
    const auto& p = input.params;
    uint32_t pixelCount = p.width * p.height;

    cl_mem voxelBuf = nullptr;
    cl_mem paramBuf = nullptr;
    cl_mem frameBuf = nullptr;
    cl_mem depthBuf = nullptr;
    cl_mem extBuf   = nullptr;
    cl_mem blurBuf  = nullptr;

    paramBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(RenderParams), (void*)&p, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"param_buffer_failed\",\"cl_error\":%d}\n", err);
        goto cleanup;
    }

    if (!input.voxels.empty()) {
        voxelBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  input.voxels.size() * sizeof(Voxel),
                                  (void*)input.voxels.data(), &err);
    } else {
        Voxel dummy = {0, 0, 0, 0};
        voxelBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(Voxel), &dummy, &err);
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"voxel_buffer_failed\",\"cl_error\":%d}\n", err);
        goto cleanup;
    }

    frameBuf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                              pixelCount * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"frame_buffer_failed\",\"cl_error\":%d}\n", err);
        goto cleanup;
    }

    depthBuf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                              pixelCount * sizeof(int32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"depth_buffer_failed\",\"cl_error\":%d}\n", err);
        goto cleanup;
    }

    extBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(ExtShaderParams), (void*)&input.extShaderParams, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "{\"error\":\"ext_buffer_failed\",\"cl_error\":%d}\n", err);
        goto cleanup;
    }

    // Clear buffers
    {
        clSetKernelArg(ctx.clearKernel, 0, sizeof(cl_mem), &frameBuf);
        clSetKernelArg(ctx.clearKernel, 1, sizeof(cl_mem), &depthBuf);
        clSetKernelArg(ctx.clearKernel, 2, sizeof(uint32_t), &pixelCount);

        size_t globalClear = ((pixelCount + 255) / 256) * 256;
        size_t localClear = 256;
        err = clEnqueueNDRangeKernel(ctx.queue, ctx.clearKernel, 1, nullptr,
                                     &globalClear, &localClear, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "{\"error\":\"clear_dispatch_failed\",\"cl_error\":%d}\n", err);
            goto cleanup;
        }
    }

    // Dispatch render kernel
    if (p.voxelCount > 0) {
        clSetKernelArg(ctx.renderKernel, 0, sizeof(cl_mem), &voxelBuf);
        clSetKernelArg(ctx.renderKernel, 1, sizeof(cl_mem), &paramBuf);
        clSetKernelArg(ctx.renderKernel, 2, sizeof(cl_mem), &frameBuf);
        clSetKernelArg(ctx.renderKernel, 3, sizeof(cl_mem), &depthBuf);
        clSetKernelArg(ctx.renderKernel, 4, sizeof(uint32_t), &p.voxelCount);
        clSetKernelArg(ctx.renderKernel, 5, sizeof(cl_mem), &extBuf);

        size_t preferredWGS = 0;
        clGetKernelWorkGroupInfo(ctx.renderKernel, ctx.device,
                                 CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                 sizeof(preferredWGS), &preferredWGS, nullptr);
        if (preferredWGS == 0) preferredWGS = 64;
        size_t localRender = preferredWGS;
        size_t globalRender = ((p.voxelCount + localRender - 1) / localRender) * localRender;

        err = clEnqueueNDRangeKernel(ctx.queue, ctx.renderKernel, 1, nullptr,
                                     &globalRender, &localRender, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "{\"error\":\"render_dispatch_failed\",\"cl_error\":%d}\n", err);
            goto cleanup;
        }
    }

    clFinish(ctx.queue);

    // Screen-space blur post-process (optional)
    if ((input.extShaderParams.flags & EXT_FLAG_BLUR) != 0u && ctx.blurKernel != nullptr) {
        blurBuf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE,
                                 pixelCount * sizeof(uint32_t), nullptr, &err);
        if (err == CL_SUCCESS) {
            clSetKernelArg(ctx.blurKernel, 0, sizeof(cl_mem), &blurBuf);
            clSetKernelArg(ctx.blurKernel, 1, sizeof(cl_mem), &frameBuf);
            clSetKernelArg(ctx.blurKernel, 2, sizeof(cl_mem), &depthBuf);
            clSetKernelArg(ctx.blurKernel, 3, sizeof(uint32_t), &p.width);
            clSetKernelArg(ctx.blurKernel, 4, sizeof(uint32_t), &p.height);
            clSetKernelArg(ctx.blurKernel, 5, sizeof(cl_mem), &extBuf);

            size_t blurGlobal = ((pixelCount + 255) / 256) * 256;
            size_t blurLocal  = 256;
            cl_int blurErr = clEnqueueNDRangeKernel(ctx.queue, ctx.blurKernel, 1, nullptr,
                                                    &blurGlobal, &blurLocal, 0, nullptr, nullptr);
            if (blurErr == CL_SUCCESS) {
                clFinish(ctx.queue);
                outputPixels.resize(pixelCount * 4);
                err = clEnqueueReadBuffer(ctx.queue, blurBuf, CL_TRUE, 0,
                                         pixelCount * sizeof(uint32_t), outputPixels.data(),
                                         0, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    fprintf(stderr, "{\"error\":\"blur_readback_failed\",\"cl_error\":%d}\n", err);
                goto cleanup;
            }
            // Blur dispatch failed — non-fatal, fall through to normal readback
            fprintf(stderr, "{\"error\":\"blur_dispatch_failed\",\"cl_error\":%d}\n", blurErr);
            err = CL_SUCCESS;
        } else {
            err = CL_SUCCESS;  // buffer alloc failed — non-fatal
        }
    }

    // Normal readback (no blur, or blur fell back)
    outputPixels.resize(pixelCount * 4);
    err = clEnqueueReadBuffer(ctx.queue, frameBuf, CL_TRUE, 0,
                              pixelCount * sizeof(uint32_t), outputPixels.data(),
                              0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        fprintf(stderr, "{\"error\":\"readback_failed\",\"cl_error\":%d}\n", err);

cleanup:
    if (blurBuf)  clReleaseMemObject(blurBuf);
    if (extBuf)   clReleaseMemObject(extBuf);
    if (depthBuf) clReleaseMemObject(depthBuf);
    if (frameBuf) clReleaseMemObject(frameBuf);
    if (voxelBuf) clReleaseMemObject(voxelBuf);
    if (paramBuf) clReleaseMemObject(paramBuf);

    return (err == CL_SUCCESS) ? 0 : EXIT_GENERIC;
}
