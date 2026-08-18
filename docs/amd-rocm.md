# AMD ROCm support

This branch supports AMD GPUs in two independent places:

1. **LoRA training** uses an AMD ROCm build of PyTorch. PyTorch exposes HIP GPUs
   through its `cuda` compatibility API, so the existing model code does not need
   a separate device name.
2. **C inference** has a correctness-first HIP/hipBLAS backend selected with
   `--backend rocm`. It offloads bf16 matrix-vector and batched matrix operations;
   operations without a ROCm implementation continue on CPU.

## Training setup

Install a mutually compatible AMD build of `torch` and `torchaudio`. Do not mix a
ROCm nightly `torch` with the generic PyPI `torchaudio` wheel. Confirm the wheel
before the GPU is installed:

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

`--mixed_precision bf16` is the default. `fp16` and `no` are diagnostic fallbacks,
not assumed to produce equivalent adapters.

## C inference on ROCm

The build expects the ROCm SDK at `/opt/rocm` and targets RDNA4 `gfx1201` by
default. Override both settings when necessary:

```bash
make rocm ROCM_PATH=/opt/rocm ROCM_ARCH=gfx1201
./qwen_tts --gpu-selftest --backend rocm
./qwen_tts -d qwen3-tts-1.7b --backend rocm -s ryan -l English \
  -t "AMD ROCm inference is active." -o output.wav
```

This initial backend is functional but not equivalent to the fused CUDA/Metal
paths: host/device synchronization still occurs around each offloaded operation.
Future performance work should keep activations and KV caches resident and port
the Talker, Code Predictor, and speech decoder kernels to HIP.
