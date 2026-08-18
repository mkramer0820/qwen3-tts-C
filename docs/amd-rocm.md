# AMD ROCm support

> **Status: implementation complete, hardware validation pending.** Nothing in
> this document is evidence of a successful R9700 build or run. The ROCm C backend,
> PyTorch training, cloning, naming, and expression workflows still require the
> hardware acceptance checklist in [ROCm review handoff](rocm-review-handoff.md).

This branch implements AMD GPU paths in two independent places:

1. **LoRA training** uses an AMD ROCm build of PyTorch. PyTorch exposes HIP GPUs
   through its `cuda` compatibility API, so the existing model code does not need
   a separate device name.
2. **C inference** has a correctness-first HIP/hipBLAS backend selected with
   `--backend rocm`. It offloads bf16 matrix-vector and batched matrix operations;
   operations without a ROCm implementation continue on CPU.

The training dependency named `qwen-tts==0.1.1` is the published Python package
that contains the Qwen3-TTS implementation and `Qwen3TTSModel`; it is not a model
generation or a replacement for the 0.6B/1.7B Qwen3-TTS checkpoints.

For a guided AMD Radeon AI PRO R9700 setup on Windows, use the separate ROCm
container profile described in [ROCm Docker on Windows/WSL2](rocm-wsl-docker.md).
It does not replace the repository's existing CUDA image or native build paths.

## Training setup

This repository's training workflow requires Linux or the supported ROCm-on-WSL
container path. Native Windows PyTorch support is a separate AMD distribution and
has not been validated with this trainer. Install matching AMD builds of `torch`
and `torchaudio` from the same ROCm source. Do not mix a ROCm nightly `torch` with
the generic PyPI `torchaudio` wheel. Then install the pinned model-facing
dependencies:

```bash
pip install -r training/expressivity-lora/requirements.txt
```

Confirm the wheel before the GPU is installed:

```bash
python training/expressivity-lora/check_rocm.py --allow-no-gpu
```

After installing the GPU and AMD driver, validate device discovery and BF16
backpropagation:

```bash
python training/expressivity-lora/check_rocm.py
```

Prepare codes and train with an explicit ROCm guard:

```bash
python training/expressivity-lora/prepare_data.py --device auto \
  --tokenizer_model_path models/tokenizer-12hz \
  --input_jsonl data/train_raw.jsonl --output_jsonl data/train_with_codes.jsonl

python training/expressivity-lora/train_lora.py \
  --require-rocm --init_model_path models/1.7B-CustomVoice \
  --train_jsonl data/train_with_codes.jsonl --output_dir out_lora
```

`--mixed_precision bf16` is the default. The checker and guarded trainer run a
real BF16 matrix multiply and backward pass, reject non-finite results, and abort
training on a non-finite loss or gradient norm. `fp16` and `no` are diagnostic
fallbacks and are not assumed to produce equivalent adapters.

## C inference on ROCm

The build expects the ROCm SDK at `/opt/rocm` and targets the locally detected GPU
(`native`) by default. Override the SDK or pass one or more explicit targets when
building on a different machine from the deployment machine:

```bash
make rocm ROCM_PATH=/opt/rocm
# Fat binary example:
make rocm ROCM_ARCH="gfx1100 gfx1101 gfx1201"
./qwen_tts --gpu-selftest --backend rocm
./qwen_tts -d qwen3-tts-1.7b --backend rocm -s ryan -l English \
  -t "AMD ROCm inference is active." -o output.wav
```

`--gpu-selftest` now exits non-zero if ROCm is unavailable; it will not validate
CPU against itself. Runtime backend selection also fails before loading the model
when ROCm is unavailable. If an individual HIP allocation, transfer, or GEMM fails
after startup, that operation is recomputed on CPU instead of consuming stale GPU
output.

This initial backend is not equivalent to the fused CUDA/Metal paths:

- Host/device synchronization occurs around every offloaded operation.
- BF16 weights are cached as FP32 device buffers, so GPU weight storage is about
  twice the source BF16 size. Leave additional VRAM for activations and the driver.
- `--int8`, `--int4`, and `--quant-mixed` kernels remain on CPU. ROCm only receives
  BF16 operations that remain in those configurations.
- `.expr` works with BF16 and INT8. It is rejected with INT4/quant-mixed until Q4
  weights can be rebuilt from the modified BF16 tensors.

Voice creation, `--load-voice`, `--voice-name`, `--icl-only`, `--instruct`, and
CLI `.expr`/`--emotion` routing are backend-independent in the code and are
intended to operate with `--backend rocm`; R9700 hardware validation is pending.
`--voice-name` is stored metadata; voices are loaded by their `.qvoice` path, not
selected by metadata name.

For a cloned voice served over HTTP, per-request `emotion` installs the steering
vector. Start the server with the appropriate language adapter when CLI-equivalent
COMBINE behavior is desired:

```bash
./qwen_tts -d qwen3-tts-1.7b --backend rocm \
  --load-voice voices/mario.qvoice --icl-only \
  --expr presets/expr/italian_csp_topk6.expr --serve 8080
```

## R9700 performance results

No performance value has been measured yet. `Undetermined` is used instead of an
estimate so later results can be traced to an actual command and output artifact.

| Test | RTF / throughput | TTFA | Peak VRAM | Result |
|---|---:|---:|---:|---|
| 0.6B preset, BF16, single stream | Undetermined | Undetermined | Undetermined | Not run |
| 1.7B preset, BF16, single stream | Undetermined | Undetermined | Undetermined | Not run |
| 1.7B cloned voice with `--icl-only` | Undetermined | Undetermined | Undetermined | Not run |
| 1.7B `.expr` plus `--emotion` | Undetermined | Undetermined | Undetermined | Not run |
| Server at 1/4/8 concurrent requests | Undetermined | Undetermined | Undetermined | Not run |
| LoRA BF16 training | Undetermined steps/s | N/A | Undetermined | Not run |

Do not compare those empty cells with the inherited Metal/CUDA tables until the
R9700 run records driver, ROCm, PyTorch, model, precision, seed, wall time, audio
duration, fallback warnings, and peak VRAM.

Future performance work should keep activations and KV caches resident and port
the Talker, Code Predictor, and speech decoder kernels to HIP.
