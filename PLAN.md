# Current roadmap

Updated: 2026-08-18

The active branch is `AMD-rocm`. Its immediate goal is to validate the opt-in
ROCm implementation on an AMD Radeon AI PRO R9700 without regressing the inherited
CPU, Metal, or CUDA paths.

## Current state

- Source review and static validation are complete through commit `df3aae4`.
- The exact Qwen3-TTS Python package contract is pinned and reviewed.
- No ROCm image, HIP build, inference run, training run, clone, expression test,
  audio-quality test, or performance benchmark has run on the R9700.
- All R9700 performance values remain `Undetermined`.

## Next gates

1. Run the Windows/WSL doctor and record driver, WSL, ROCm, ROCDXG, Torch, HIP,
   detected device, and `gfx` target.
2. Build both Docker profiles; run the ROCm BF16 probe and C GPU self-test.
3. Exercise 0.6B and 1.7B preset inference plus forced CPU fallback.
4. Test cloned `.qvoice`, `--voice-name`, `--icl-only`, `.expr`, emotion, and HTTP
   behavior using the acceptance checklist.
5. Run a short guarded LoRA job, save/resume it, export `.expr`, and audition it.
6. Record raw commands, logs, audio artifacts, RTF, TTFA, and peak VRAM before
   changing any status or performance table from `Not run`/`Undetermined`.

The detailed checklist and known risks live in
[docs/rocm-review-handoff.md](docs/rocm-review-handoff.md). Historical roadmaps and
experiments are indexed under [docs/archive](docs/archive/README.md).
