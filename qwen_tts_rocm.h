/* qwen_tts_rocm.h - AMD ROCm/HIP backend, C-callable surface. */
#ifndef QWEN_TTS_ROCM_H
#define QWEN_TTS_ROCM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int   qwen_rocm_available(void);
void *qwen_rocm_init(void);
void  qwen_rocm_free(void *ctx);
/* Non-zero tells the backend wrapper to execute the CPU implementation. */
int   qwen_rocm_matvec_bf16(void *ctx, float *y, const uint16_t *W,
                            const float *x, int rows, int cols);
int   qwen_rocm_matmat_bf16(void *ctx, float *Y, const uint16_t *W,
                            const float *X, int rows, int cols, int B);

#ifdef __cplusplus
}
#endif
#endif
