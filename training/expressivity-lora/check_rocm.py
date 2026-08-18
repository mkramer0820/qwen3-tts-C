#!/usr/bin/env python3
"""Fail-fast ROCm validation for Qwen3-TTS LoRA training."""
import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048, help="BF16 square matmul size")
    parser.add_argument("--allow-no-gpu", action="store_true",
                        help="only validate that this is a HIP wheel when hardware is not installed yet")
    args = parser.parse_args()

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

    props = torch.cuda.get_device_properties(0)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"properties={props}")
    arch = getattr(props, "gcnArchName", "unknown")
    print(f"arch={arch}")

    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = (a @ b).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(a.grad).all() or not torch.isfinite(b.grad).all():
        print("FAIL: non-finite BF16 result or gradient", file=sys.stderr)
        return 3
    print(f"PASS: BF16 forward/backward (loss={loss.item():.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
