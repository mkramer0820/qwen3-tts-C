#!/usr/bin/env python3
"""Fail-fast ROCm validation for Qwen3-TTS LoRA training."""
import argparse
import sys

import torch

from rocm_validation import (
    check_training_packages,
    require_rocm_device,
    run_bf16_forward_backward,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048, help="BF16 square matmul size")
    parser.add_argument("--allow-no-gpu", action="store_true",
                        help="only validate that this is a HIP wheel when hardware is not installed yet")
    args = parser.parse_args()

    try:
        packages = check_training_packages(require_rocm_stack=True)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 5
    print("packages=" + ", ".join(f"{name}=={value}" for name, value in packages.items()))
    print(f"torch={torch.__version__}")
    print(f"hip={torch.version.hip}")
    if torch.version.hip is None:
        print("FAIL: this is not a ROCm/HIP PyTorch build", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        if args.allow_no_gpu:
            print("PASS: ROCm wheel imports; GPU validation deferred")
            return 0
        print("FAIL: ROCm PyTorch imports, but no GPU is visible", file=sys.stderr)
        return 2

    try:
        props = require_rocm_device(require_bf16=True)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 3
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"properties={props}")
    arch = getattr(props, "gcnArchName", "unknown")
    print(f"arch={arch}")

    try:
        loss = run_bf16_forward_backward(args.size)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    print(f"PASS: BF16 forward/backward (loss={loss:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
