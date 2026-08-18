# ROCm Docker on Windows/WSL2

> **Status: configuration only; not yet run on the target machine.** Compose and
> PowerShell syntax validate, but the image has not been built and GPU access has
> not been tested because Ubuntu WSL and the R9700 runtime are not installed yet.

This is the guided Windows path for an **AMD Radeon AI PRO R9700**. The card is
an RDNA4 `gfx1201` device and is supported by AMD's ROCm 7.2.1 ROCDXG WSL stack.

The versions in this guide were checked on 2026-08-18 against AMD's
[ROCDXG WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html),
[ROCDXG compatibility table](https://github.com/ROCm/librocdxg), and
[ROCm PyTorch container](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/native_linux/install-pytorch.html)
instructions.

The repository remains cross-platform. `compose.yaml` provides separate `cuda`
and `rocm` profiles. The CUDA image shares the pinned Qwen3-TTS model dependencies
and verifies that CUDA-enabled Torch survives installation; ROCm host/runtime
requirements remain isolated to the AMD profile. Native CPU, CUDA, Metal, and ROCm
builds continue to use separate Makefile targets.

## What you need on Windows

- Windows 11 with virtualization enabled.
- AMD Software: Adrenalin Edition 26.2.2 for WSL2 (the production ROCDXG pair
  documented for ROCm 7.2.1).
- PowerShell. Run as Administrator only when Windows asks for elevation.
- Free disk space for Ubuntu, ROCm, the container, Python packages, and models.

`docker-desktop` in `wsl --list --verbose` is not a normal Ubuntu environment.
The setup script installs Ubuntu 24.04 and Docker Engine inside that distribution,
which is the layout used by AMD's WSL container instructions.
If Docker Desktop integration injects its daemon into Ubuntu, the setup exits with
instructions to disable that integration for this distribution.

## First-time setup

Open PowerShell in the repository and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\rocm-wsl.ps1
```

The script performs these steps and stops at the first actionable failure:

1. Confirms Windows reports an AMD Radeon AI PRO R9700.
2. Installs `Ubuntu-24.04` if it is absent and ensures it uses WSL2.
3. Checks `/dev/dxg`, verifies any installed ROCm/ROCDXG package versions, then
   installs ROCm 7.2.1 with AMD's WSL use case and librocdxg 1.2.0 when absent.
   Mismatched versions stop rather than being silently accepted.
4. Installs and starts Docker Engine plus Compose inside Ubuntu.
5. SHA-256 verifies the two pinned installer packages and builds the official
   ROCm 7.2.1/PyTorch 2.9.1 based image with exact model-facing Python versions.
6. Selects the R9700 by device name when multiple AMD GPUs are present, then runs
   PyTorch ROCm/BF16 checks and reports the selected device's `gfx` target.

An initial Ubuntu installation may require a Windows restart and one manual launch
of Ubuntu to choose a Linux username and password. Rerun the same command afterward.
The Linux password is required for `sudo`; it is not your Windows PIN.

## Build and verify C inference

```powershell
.\rocm-wsl.ps1 -Action build
```

This detects `gfx1201`, runs `make rocm`, and requires the GPU self-test to pass.
Other useful actions are:

```powershell
.\rocm-wsl.ps1 -Action doctor
.\rocm-wsl.ps1 -Action shell
```

Inside the container shell, the repository is `/workspace/qwen3-tts-C`. Models,
voices, expressions, and output files remain in the Windows repository because it
is bind-mounted rather than copied into the image.

## Inference, cloning, and expression

From `-Action shell`:

```bash
# Download the repository's expression assets once
bash download_assets.sh

# Preset voice and expression
qwen-rocm infer -d qwen3-tts-1.7b -s ryan -l Italian \
  --expr presets/expr/italian_csp_topk6.expr --emotion joy \
  --text "Il contenitore ROCm funziona." -o joy.wav

# Save a named clone (the name is metadata; the file path selects the voice)
qwen-rocm infer -d qwen3-tts-1.7b-base --ref-audio reference.wav -l Italian \
  --voice-name "My Voice" --save-voice voices/my-voice.qvoice \
  --text "Questa e la mia voce riutilizzabile." -o clone-check.wav

# Use that clone with a 1.7B expression adapter
qwen-rocm infer -d qwen3-tts-1.7b --load-voice voices/my-voice.qvoice \
  --icl-only -l Italian --expr presets/expr/italian_csp_topk6.expr --emotion joy \
  --text "Questa usa la mia voce clonata." -o clone-joy.wav
```

`--voice-name` does not create a registry key; load the `.qvoice` by path. For a
cloned HTTP voice, provide `--expr` when starting the server to get the same
COMBINE behavior as CLI emotion. INT4/quant-mixed expression adapters remain
unsupported; use BF16 or INT8.

## Training

From the container shell, prepare data normally, then use the guarded command:

```bash
qwen-rocm train \
  --init_model_path models/1.7B-CustomVoice \
  --train_jsonl training/expressivity-lora/data/train_with_codes.jsonl \
  --output_dir training/expressivity-lora/out_lora
```

The wrapper always adds `--require-rocm`, so training cannot silently run on CPU
or a non-ROCm PyTorch build. It verifies the pinned Qwen/Transformers/PEFT stack,
runs a BF16 forward/backward probe, and aborts on non-finite training values.

## Hardware scope

This workflow deliberately accepts RDNA3/RDNA4 targets (`gfx11*` and `gfx12*`).
The primary acceptance target is the AMD Radeon AI PRO R9700 (`gfx1201`). AMD
Instinct/CDNA/MI-series accelerators are out of scope for this repository review;
the launcher rejects them instead of implying they were tested.

## Direct Compose use

Advanced users can bypass PowerShell from Ubuntu WSL:

```bash
docker compose --profile rocm build rocm-wsl
docker compose --profile rocm run --rm rocm-wsl doctor
docker compose --profile rocm run --rm rocm-wsl build
docker compose --profile rocm run --rm rocm-wsl shell
```

For NVIDIA systems, the existing image remains available through:

```bash
docker compose --profile cuda build cuda
docker compose --profile cuda run --rm cuda bash
```
