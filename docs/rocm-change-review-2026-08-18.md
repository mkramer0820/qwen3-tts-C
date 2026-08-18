# ROCm compatibility changes for independent review

Date: 2026-08-18

Purpose: give the next reviewer (including Claude) enough context to confirm,
challenge, or revise the ROCm changes without relying on chat history.

For a standalone set of review instructions, use
`docs/claude-review-prompt.md`.

## Validation status: NOT HARDWARE TESTED

Do not approve these changes because syntax/configuration checks pass. No ROCm
Docker image was built, no HIP source was compiled, and no command was run on the
AMD Radeon AI PRO R9700. Training, voice cloning, voice naming/metadata, expression
adapters, HTTP emotion behavior, numerical parity, fallback behavior, and audible
output all remain unverified on hardware. The acceptance checklist at the end is
required before describing this branch as working or production-ready.

## Upstream synchronization included in this review

The branch now merges 34 commits from `gabriele-mastrapasqua/qwen3-tts:main`
through commit `328ab9c` (`v0.19.2-1-g328ab9c`). The merge brings in the Ingot
safetensors/GGUF library migration, the fused GPU Talker server replay fix,
AVX-512 parity work, CI action updates, and the complete 0.6B emotion asset set.

The merge exposed one ROCm-specific build issue: upstream added `libingot.a` to
the CPU, Metal, and CUDA link commands, but the fork's ROCm target did not exist
upstream and therefore was not updated. `rocm_build` now depends on and links
`$(INGOT_LIB)`. The README was the only textual merge conflict. Full details and
independent review points are in `docs/upstream-sync-2026-08-18.md`.

## Problems addressed

1. A requested GPU backend could fall back to CPU, after which `--gpu-selftest`
   compared CPU with CPU and returned PASS.
2. HIP allocation, transfer, or hipBLAS failures returned without producing a
   valid output tensor; synthesis then consumed stale/uninitialized output.
3. `make rocm` defaulted to `gfx1201`, producing an incompatible binary for most
   other AMD GPUs unless the user knew to override it.
4. Expressivity LoRA training omitted `talker.text_projection` and used a label
   shift that conflicts with modern Transformers causal-LM loss behavior.
5. Training accepted malformed codec tensors and BF16-unsupported devices until
   much later failures.
6. INT4 expression adapters and cloned-voice HTTP emotion behavior were presented
   more broadly than the implementation supports.

## Code changes

- `qwen_tts_rocm.{h,cpp}` now return operation status. Device-buffer growth keeps
  the old allocation when a larger allocation fails, cached weights include their
  element count, and every HIP failure returns an error.
- `qwen_tts_kernels.{h,c}` expose explicit BF16 CPU entry points. They bypass the
  global GPU hook without changing global state, allowing safe per-operation CPU
  fallback without recursion or cross-thread hook races.
- `qwen_tts_backend.c` recomputes a failed ROCm matvec/matmat on CPU. GPU self-test
  returns failure when the requested backend is unavailable.
- `main.c` rejects unknown backends, checks backend availability before model load,
  rejects `.expr` with INT4/quant-mixed, records whether an expression adapter was
  applied, and warns that ROCm does not accelerate quantized kernels.
- `Makefile` defaults `ROCM_ARCH` to `native`; a space-separated list creates an
  RDNA3/RDNA4 fat binary, for example `ROCM_ARCH="gfx1100 gfx1101 gfx1201"`.
- `training/expressivity-lora/train_lora.py` applies `talker.text_projection`, uses
  unshifted main-talker labels, aligns sub-talker hidden states as `hidden[:-1]`
  with `codec_mask[1:]`, and fails early for unsupported BF16/API combinations.
- `training/expressivity-lora/dataset.py` requires `audio_codes` shape `[frames,16]`.
- `training/expressivity-lora/check_rocm.py` now checks reported BF16 support.
- `qwen_tts_server.c` reports when cloned-voice HTTP emotion is STEER-only because
  no startup `.expr` was applied.
- `docs/amd-rocm.md` and the new training requirements file document installation,
  architecture selection, VRAM expansion, quantization, voice naming, cloning,
  and server expression behavior.

## Points the reviewer should challenge

1. **Training label alignment:** the change follows modern Transformers causal-LM
   semantics and Qwen3-TTS PR #278. Verify it against the exact `qwen-tts` and
   `transformers` versions selected for deployment. If an older model implementation
   expects pre-shifted labels, pin versions or compute the loss explicitly rather
   than reverting silently.
2. **BF16 capability guard:** confirm `torch.cuda.is_bf16_supported()` is reliable
   for every intended ROCm GPU. If a supported AMD stack reports a false negative,
   replace this with a small forward/backward probe shared with `check_rocm.py`.
3. **CPU fallback policy:** fallback preserves correctness but a repeatedly failing
   GPU can become noisy and slow. Consider disabling ROCm after the first operation
   failure or making strict failure selectable by environment variable.
4. **FP32 device weight cache:** this patch documents but does not redesign it.
   A future HIP/hipBLASLt implementation should store BF16 weights directly after
   hardware validation and numerical comparison.
5. **HTTP clone expressions:** adapters mutate backbone weights and worker clones
   share those weights, so changing language `.expr` per request is not currently
   safe. The implemented behavior requires a startup `--expr`. A per-request design
   needs immutable adapter-aware kernels or independent weight copies.
6. **Backend lifetime:** the process-owned GPU backend remains intentionally alive
   until process exit. A later cleanup refactor should uninstall global hooks before
   freeing it and cover every early-return path.

## Verification completed here

- `git diff --check` and Python AST parsing.
- Source-level review of hook bypass/fallback paths and model/voice/expression order.
- Default non-ROCm compilation where the available host toolchain permits it.

Not completed here: HIP compilation, `--gpu-selftest` on AMD hardware, ROCm
forward/backward training, model-based voice cloning, or audible expression tests.
Those require a supported AMD GPU, matching ROCm SDK, model files, and Python
training dependencies.

## Docker/WSL additions for review

- `training/expressivity-lora/docker/Dockerfile.rocm` extends AMD's pinned ROCm
  7.2.1/PyTorch 2.9.1 image. The original CUDA Dockerfile remains unchanged.
- `compose.yaml` exposes separate opt-in `rocm` and `cuda` profiles. The ROCm WSL
  service uses AMD's documented `/dev/dxg`, `libdxcore`, `librocdxg`, device-ID
  configuration mounts, and `HSA_ENABLE_DXG_DETECTION=1`.
- `rocm-wsl.ps1` installs/selects a real Ubuntu 24.04 WSL2 distribution;
  it does not mistake Docker Desktop's internal distribution for a Linux workspace.
- `training/expressivity-lora/docker/rocm-wsl.sh` installs ROCm 7.2.1, librocdxg
  1.2.0, and Ubuntu Docker Engine, then builds and checks the container.
- `rocm-entrypoint.sh` prefers the R9700 by PyTorch device name, isolates it with
  `HIP_VISIBLE_DEVICES`, derives its `gfx1201` target from `gcnArchName`, checks
  PyTorch/BF16, compiles for that target, and runs the strict C GPU self-test.

Reviewer caveats: confirm AMD's pinned driver/container pair is still current at
deployment time; test Docker Compose device/mount behavior on the target Windows
host; and decide whether downloading/installing the versioned AMD host package
should remain automated or become an explicitly version-pinned manual prerequisite.

### Requested review scope

Review and hardware acceptance are limited to RDNA4 and RDNA3 (`gfx12*`/`gfx11*`).
The primary machine is an AMD Radeon AI PRO R9700, RDNA4 `gfx1201`, with 32 GiB
VRAM. AMD Instinct/CDNA/MI-series support is explicitly out of scope; do not block
the R9700 work on MI accelerator portability.

Claude should independently inspect and either confirm or dispute each item:

1. PowerShell handling of distro names, NUL-padded `wsl.exe` output, paths with
   spaces/OneDrive, first-launch/reboot behavior, and WSL1-to-WSL2 conversion.
2. Whether AMD still supports exactly the pinned ROCm 7.2.1, librocdxg 1.2.0,
   PyTorch 2.9.1 image, and Adrenalin 26.2.2 production combination.
3. Whether `/dev/dxg`, `libdxcore.so`, `librocdxg.so`, `dids.conf`, and the HSA
   environment variable match AMD's current ROCDXG container contract and work
   with Docker Engine inside Ubuntu; verify Docker Desktop detection as well.
4. Whether PyTorch device-name selection plus `HIP_VISIBLE_DEVICES` reliably picks
   the R9700 ahead of an integrated Radeon, and whether `gcnArchName` remains the
   correct source for the `hipcc --offload-arch` target on RDNA3/RDNA4.
5. Whether the image preserves the ROCm PyTorch wheel when installing `qwen-tts`
   and paired audio/training dependencies; propose pins if dependency resolution
   can replace it with a non-ROCm wheel.
6. C backend correctness under `gfx1201`: operation-failure CPU recomputation,
   strict backend/self-test behavior, BF16 numerical tolerance, and thread safety.
7. LoRA training semantics: text projection, label shifts, codec alignment, BF16
   forward/backward, save/resume, and `.expr` export against the installed versions.
8. Voice workflow behavior on ROCm: clone creation, `.qvoice` reuse, metadata-only
   `--voice-name`, `--icl-only`, startup versus request-time expression handling,
   and the documented INT4/quant-mixed restriction.
9. Global repository compatibility: CUDA image/profile remains functional, native
   CPU/CUDA/Metal builds are unaffected, and no AMD default leaks into other users.
   Confirm that every native target, including ROCm, links the vendored Ingot
   library after removal of the legacy safetensors reader.
10. Usability from a clean Windows 11 host that initially lists only
    `docker-desktop`: every failure should state the next corrective action.
11. Populate the README and `docs/amd-rocm.md` R9700 performance tables with raw
    command evidence. Leave values `Undetermined` rather than extrapolating from
    CUDA, Metal, a different Radeon GPU, or theoretical bandwidth.

## Hardware acceptance checklist

1. Build with `make rocm`; confirm the reported target matches `rocminfo`.
2. Run `./qwen_tts --gpu-selftest --backend rocm`; unplug/hide the GPU and confirm
   the same command exits non-zero rather than PASS.
3. Generate BF16 preset, cloned `.qvoice`, cloned `--emotion`, and manual `--expr`
   samples; compare against CPU with identical seeds.
4. Inject or simulate an allocation failure and confirm output is recomputed on CPU.
5. Run `check_rocm.py`, a short LoRA training epoch, export `.expr`, and audition it.
6. Start a cloned-voice server both with and without startup `--expr`; confirm the
   warning and documented STEER versus COMBINE behavior.
