/*
 * qwen_tts_rocm.cpp - AMD ROCm backend using HIP and hipBLAS.
 *
 * This is the correctness-first backend: bf16 model weights are converted once
 * on the host, cached as resident fp32 device buffers, and multiplied by hipBLAS.
 * Unsupported operations continue through the existing CPU implementation.
 */
#include "qwen_tts_rocm.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>

struct weight_cache_entry { const void *key; float *device; };
struct qwen_rocm_ctx {
    hipblasHandle_t handle;
    weight_cache_entry *weights;
    int count, capacity;
    float *dx, *dy;
    size_t dx_bytes, dy_bytes;
};

static float bf16_to_f32(uint16_t value) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)value << 16;
    return v.f;
}

extern "C" int qwen_rocm_available(void) {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

extern "C" void *qwen_rocm_init(void) {
    qwen_rocm_ctx *ctx = (qwen_rocm_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) return nullptr;
    if (hipblasCreate(&ctx->handle) != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ROCm: hipblasCreate failed\n");
        free(ctx);
        return nullptr;
    }
    return ctx;
}

extern "C" void qwen_rocm_free(void *opaque) {
    qwen_rocm_ctx *ctx = (qwen_rocm_ctx *)opaque;
    if (!ctx) return;
    for (int i = 0; i < ctx->count; ++i) hipFree(ctx->weights[i].device);
    free(ctx->weights);
    if (ctx->dx) hipFree(ctx->dx);
    if (ctx->dy) hipFree(ctx->dy);
    if (ctx->handle) hipblasDestroy(ctx->handle);
    free(ctx);
}

static float *resident_weight(qwen_rocm_ctx *ctx, const uint16_t *weight, size_t count) {
    for (int i = 0; i < ctx->count; ++i)
        if (ctx->weights[i].key == weight) return ctx->weights[i].device;

    float *host = (float *)malloc(count * sizeof(float));
    if (!host) return nullptr;
    for (size_t i = 0; i < count; ++i) host[i] = bf16_to_f32(weight[i]);
    float *device = nullptr;
    if (hipMalloc((void **)&device, count * sizeof(float)) != hipSuccess ||
        hipMemcpy(device, host, count * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
        if (device) hipFree(device);
        free(host);
        return nullptr;
    }
    free(host);

    if (ctx->count == ctx->capacity) {
        int next = ctx->capacity ? ctx->capacity * 2 : 64;
        void *grown = realloc(ctx->weights, (size_t)next * sizeof(*ctx->weights));
        if (!grown) { hipFree(device); return nullptr; }
        ctx->weights = (weight_cache_entry *)grown;
        ctx->capacity = next;
    }
    ctx->weights[ctx->count++] = {weight, device};
    return device;
}

static float *grow_device(float **buffer, size_t *capacity, size_t bytes) {
    if (*capacity >= bytes) return *buffer;
    if (*buffer) hipFree(*buffer);
    *buffer = nullptr; *capacity = 0;
    if (hipMalloc((void **)buffer, bytes) != hipSuccess) return nullptr;
    *capacity = bytes;
    return *buffer;
}

extern "C" void qwen_rocm_matmat_bf16(void *opaque, float *output,
                                         const uint16_t *weight, const float *input,
                                         int rows, int cols, int batch) {
    qwen_rocm_ctx *ctx = (qwen_rocm_ctx *)opaque;
    float *dw = resident_weight(ctx, weight, (size_t)rows * cols);
    float *dx = grow_device(&ctx->dx, &ctx->dx_bytes, (size_t)cols * batch * sizeof(float));
    float *dy = grow_device(&ctx->dy, &ctx->dy_bytes, (size_t)rows * batch * sizeof(float));
    if (!dw || !dx || !dy) { fprintf(stderr, "ROCm: device allocation failed\n"); return; }
    if (hipMemcpy(dx, input, (size_t)cols * batch * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "ROCm: input upload failed\n"); return;
    }
    const float alpha = 1.0f, beta = 0.0f;
    hipblasStatus_t status = hipblasSgemm(ctx->handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                          batch, rows, cols, &alpha,
                                          dx, batch, dw, cols, &beta, dy, batch);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ROCm: hipblasSgemm failed (%d)\n", (int)status);
        return;
    }
    if (hipMemcpy(output, dy, (size_t)rows * batch * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
        fprintf(stderr, "ROCm: output download failed\n");
}

extern "C" void qwen_rocm_matvec_bf16(void *ctx, float *output,
                                         const uint16_t *weight, const float *input,
                                         int rows, int cols) {
    qwen_rocm_matmat_bf16(ctx, output, weight, input, rows, cols, 1);
}
