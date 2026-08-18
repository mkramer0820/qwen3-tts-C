# ROCm review handoff

Updated: 2026-08-18

This is the current source of truth for independent review of the AMD ROCm fork.
It replaces the older dated change-review and upstream-sync notes. The final
source-review follow-up is commit `df3aae4`.

## Status

**Not hardware tested.** No ROCm image has been built on the target machine, no
HIP source has been compiled for the Radeon AI PRO R9700, and no R9700 inference,
training, cloning, expression, audio-quality, or performance result exists yet.
Static validation does not change that status.

Two independent review passes are complete. The second reviewer retracted its
training-semantics finding after inspecting the exact `qwen-tts==0.1.1` and
`transformers==4.57.3` wheels. This closes the source finding, not the hardware
acceptance work.

Primary target: AMD Radeon AI PRO R9700, RDNA4 `gfx1201`, 32 GiB, Windows 11 with
Ubuntu 24.04 WSL2. Review scope also includes RDNA3/RDNA4 `gfx11*` and `gfx12*`.
MI/CDNA accelerators are not an acceptance target.

## History boundary

- Fork branch: `AMD-rocm`
- Upstream: `gabriele-mastrapasqua/qwen3-tts:main`
- Upstream merged through: `328ab9c` (`v0.19.2-1-g328ab9c`)
- Reviewed merge commit: `7a74a0a`
- `c950a3a` accidentally deleted 87 newly merged files.
- `e99b3d5` restores exactly those files; the two commits cancel each other.
- `b05a1bb` adds the standalone review prompt only.
- `a9bdda0` implements the review hardening and documentation consolidation.
- `df3aae4` addresses final review follow-ups and is the reviewed static baseline.

The 34-commit upstream merge includes the Ingot safetensors/GGUF migration, fused
GPU Talker server replay fix, AVX-512 parity work, CI updates, and all 0.6B emotion
assets and cloned-voice directions. `README.md` was the only textual conflict.

## Implemented ROCm behavior

- HIP/hipBLAS BF16 matvec and matmat offload selected by `--backend rocm`.
- Strict backend availability and GPU self-test; unavailable ROCm cannot pass by
  comparing CPU output against CPU output.
- Explicit CPU entry points for per-operation fallback without GPU-hook recursion.
- FP32 resident device-weight cache with shape validation.
- Mutex-protected ROCm cache, shared transfer buffers, and hipBLAS handle.
- `make rocm` links the vendored Ingot library after removal of the legacy reader.
- `ROCM_ARCH=native` by default, with explicit multi-target RDNA builds supported.
- Separate opt-in ROCm and CUDA Compose profiles.

## Training and container hardening

- The model-facing stack is pinned to `qwen-tts==0.1.1`,
  `transformers==4.57.3`, `accelerate==1.12.0`, and `peft==0.18.1`.
- `qwen-tts` is the distribution name for the Qwen3-TTS Python implementation;
  `0.1.1` is its package release, not the model generation.
- In the ROCm workflow, Torch and torchaudio must remain matching 2.9.1 builds;
  local ROCm version suffixes are accepted while mismatched base versions fail.
- CUDA/CPU training still enforces the shared model API pins but does not inherit
  the ROCm container's Torch/torchaudio version contract.
- Container build validation checks those versions, required imports, PEFT layer
  selection parameters, and preservation of ROCm PyTorch 2.9.1.
- `check_rocm.py` and guarded training share package, device, and BF16 probes.
- Guarded BF16 training executes a real forward/backward probe before model load.
- Training reduces finite-state checks across Accelerator workers and aborts all
  workers on a non-finite loss or synchronized gradient norm.
- The WSL installer verifies installed `rocm-core` and `rocdxg-roct` versions.
- Host installation uses AMD's documented `--usecase=wsl,rocm --no-dkms` path.
- ROCm 7.2.x remains paired with librocdxg 1.2.0 per AMD's compatibility table;
  a newer ROCDXG tag is not adopted without compatibility and hardware evidence.
- Downloaded AMD and ROCDXG installer packages are SHA-256 verified before install.
- Git attributes keep shell scripts LF-only when used from a Windows checkout.
- The existing CUDA training image now installs the shared pinned model stack;
  ROCm remains an additional Compose profile rather than replacing CUDA.

## Reviewed findings disposition

1. **ROCDXG version:** keep 1.2.0 for ROCm 7.2.x; verify rather than silently
   accepting another version. Re-evaluate only with AMD compatibility evidence.
2. **ROCm context concurrency:** fixed with a portable C++ mutex and proper C++
   construction/destruction of the context.
3. **BF16 and numerical validation:** fixed with a shared execution probe plus
   finite loss and gradient checks.
4. **Host package detection:** fixed; package versions are checked, not just files.
5. **PowerShell `$linuxRepo`:** defensively quoted. PowerShell already passed the
   variable as one argument, but the explicit quoting protects future refactors.
6. **Package integrity:** fixed for the two directly downloaded `.deb` artifacts.
7. **Python dependency drift:** fixed for the model-facing packages whose private
   APIs and label semantics the trainer depends on.

## Remaining review risks

- Confirm the pinned package set against a real short training run. Exact-wheel
  inspection confirms qwen-tts 0.1.1 has separate `text_embedding` and
  `text_projection` modules, while Transformers 4.57.3 `ForCausalLMLoss` shifts
  unshifted labels internally. A real run must still confirm output quality.
- Confirm the R9700 reports `gfx1201`, BF16 works, and device-name selection wins
  over any integrated Radeon adapter.
- Measure the FP32 weight cache's actual VRAM usage for 0.6B and 1.7B models.
- Exercise allocation/copy/GEMM failure fallback and inspect resulting audio.
- Validate Docker Engine inside Ubuntu WSL, including all ROCDXG mounts.
- Test cloned `.qvoice`, metadata-only `--voice-name`, `--icl-only`, 0.6B emotion,
  1.7B `.expr`, HTTP emotion behavior, and INT4 restrictions.

## Hardware acceptance checklist

1. Record Windows driver, Ubuntu, ROCm, ROCDXG, Torch, HIP, GPU name, and `gfx`.
2. Build with `make rocm` and confirm `third_party/ingot/libingot.a` is linked.
3. Run `./qwen_tts --gpu-selftest --backend rocm`; hide the GPU and verify failure.
4. Generate matched CPU/ROCm preset samples with fixed seeds.
5. Generate and reload a named clone, then test `--icl-only` and expression output.
6. Test all 0.6B preset emotion assets and a cloned-voice emotion direction.
7. Start the HTTP server with and without startup `.expr`; verify documented mode.
8. Run `check_rocm.py`, a short guarded LoRA job, resume/save, export `.expr`, and
   audition the result. Confirm loss and gradient values remain finite.
9. Record RTF, TTFA, peak VRAM, fallback warnings, commands, and output artifacts.
10. Populate performance tables only from those recorded R9700 results.

## Evidence required for approval

Report findings first with file and line references. Separate static inspection,
compilation, container execution, and R9700 hardware evidence. An approval based
only on syntax checks, documentation, or inferred hardware behavior is invalid.
