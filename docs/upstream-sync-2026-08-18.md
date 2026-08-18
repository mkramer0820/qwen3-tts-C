# Upstream synchronization review

Date: 2026-08-18

This document records the merge of `gabriele-mastrapasqua/qwen3-tts:main` into
the `AMD-rocm` branch so another reviewer can confirm or dispute the result
without relying on chat history.

## Merge boundary

- Previous common ancestor: `36f05cd`
- Upstream tip merged: `328ab9c`
- Upstream description: `v0.19.2-1-g328ab9c`
- Upstream-only commits: 34
- Fork-only commits before the merge: 4
- Textual conflicts: `README.md` only

The README resolution preserves the fork's explicit ROCm status, R9700 quick
start, validation matrix, and undetermined performance values while retaining
upstream's latest AVX-512, CUDA/Metal, and 0.6B emotion documentation.

## Upstream changes brought forward

1. The legacy `qwen_tts_safetensors` reader is replaced by the vendored Ingot
   v1 library. Ingot supplies safetensors/GGUF reading and quantization support.
2. The fused GPU Talker server path clears stale delta-prefill KV state so a new
   request does not replay data from the previous request.
3. AVX-512 work adds BF16 dot-product, VNNI, fused-QKV, attention, RMS, and
   convolution parity improvements plus `-mavx512dq` where required.
4. The 0.6B model gains all six emotion assets for all nine presets, generic
   emotion directions for cloned voices, demos, and supporting documentation.
5. GitHub Actions dependencies and compiler include paths are updated for Ingot.

## ROCm compatibility decisions

- `rocm_build` now depends on and links `third_party/ingot/libingot.a`. Without
  this fork-specific fix, the merged ROCm binary would fail to link after the
  legacy safetensors reader was removed.
- The Ingot source is present in the normal Docker build context; `.dockerignore`
  does not exclude `third_party`.
- The ROCm backend, strict backend selection, per-operation CPU fallback, and
  `gfx11*`/`gfx12*` WSL workflow remain present after the merge.
- Upstream's CPU and AVX-512 kernel changes coexist with the fork's explicit
  `qwen_matvec_bf16_cpu` and `qwen_matmat_bf16_cpu` fallback entry points.
- The 0.6B emotion changes are backend-independent. They should work on ROCm
  when model execution works, but no R9700 audio or performance claim is made.
- CUDA, Metal, and CPU remain first-class inherited targets. ROCm remains opt-in
  and does not become the default build or container profile.

## Verification performed

- Trial merge and conflict-marker scan.
- `git diff --check` after the ROCm/Ingot correction.
- Static inspection of Makefile target dependencies and Docker build context.
- Static inspection of backend selection, CPU fallback, voice-clone expression
  warning, and merged 0.6B emotion paths.
- PowerShell parse check, Python syntax/AST checks, and Compose schema validation
  where the host tools are available.

No supported Linux C/HIP compiler, running Docker daemon, model files, or R9700
runtime was available in this review environment. Therefore HIP compilation,
model loading through Ingot, C GPU self-test, inference, cloning, expression,
training, and audio-quality checks remain pending.

## Independent reviewer checklist

1. Build `make blas`, `make metal`, `make cuda`, and `make rocm` on their native
   platforms and confirm every target links Ingot exactly once.
2. Run Ingot's test suite and load both 0.6B and 1.7B safetensors checkpoints.
3. On the R9700, run the Docker doctor/build flow and the strict ROCm self-test.
4. Generate matched CPU/ROCm preset, clone, 0.6B emotion, 1.7B `.expr`, and HTTP
   server samples with fixed seeds; inspect logs and audition the outputs.
5. Reproduce the server replay scenario fixed by upstream issue 19 on all fused
   GPU backends available to the reviewer.
6. Confirm upstream AVX-512 changes do not affect the explicit CPU fallback used
   after a ROCm operation failure.
7. Leave all R9700 performance cells `Undetermined` until raw commands and output
   files from the target hardware are attached.
