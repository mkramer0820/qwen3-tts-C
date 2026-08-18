"""Shared dependency and ROCm execution checks for expressivity training."""

from importlib.metadata import PackageNotFoundError, version
import inspect

import torch


EXPECTED_PACKAGES = {
    "qwen-tts": "0.1.1",
    "transformers": "4.57.3",
    "accelerate": "1.12.0",
    "peft": "0.18.1",
}
EXPECTED_VERSION_PREFIXES = {
    "torch": "2.9.1",
    "torchaudio": "2.9.1",
}


def check_training_packages(require_rocm_stack=False):
    """Fail when model-facing package versions or required APIs drift."""
    problems = []
    installed_versions = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            problems.append(f"{package} is not installed")
            continue
        installed_versions[package] = installed
        if installed != expected:
            problems.append(f"{package}=={installed}, expected {expected}")
    if require_rocm_stack:
        for package, expected_prefix in EXPECTED_VERSION_PREFIXES.items():
            try:
                installed = version(package)
            except PackageNotFoundError:
                problems.append(f"{package} is not installed")
                continue
            installed_versions[package] = installed
            if not installed.startswith(expected_prefix):
                problems.append(f"{package}=={installed}, expected {expected_prefix}.*")

    if problems:
        raise RuntimeError("incompatible training environment: " + "; ".join(problems))

    try:
        from accelerate import Accelerator  # noqa: F401
        from peft import LoraConfig
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: F401
        import torchaudio  # noqa: F401
        from transformers import AutoConfig  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"pinned training package import failed: {exc}") from exc

    lora_parameters = inspect.signature(LoraConfig).parameters
    for required in ("layers_to_transform", "layers_pattern"):
        if required not in lora_parameters:
            raise RuntimeError(f"peft LoraConfig is missing required parameter {required}")

    return installed_versions


def require_rocm_device(require_bf16=True):
    """Return selected-device properties or raise with an actionable error."""
    if torch.version.hip is None:
        raise RuntimeError("this is not a ROCm/HIP PyTorch build (torch.version.hip is None)")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch imports, but no GPU is visible")
    if require_bf16 and not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        raise RuntimeError("this GPU/runtime does not report BF16 support")
    return torch.cuda.get_device_properties(0)


def run_bf16_forward_backward(size=512):
    """Exercise BF16 allocation, GEMM, backward, synchronization, and finiteness."""
    require_rocm_device(require_bf16=True)
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = (a @ b).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    finite = (
        torch.isfinite(loss).item()
        and torch.isfinite(a.grad).all().item()
        and torch.isfinite(b.grad).all().item()
    )
    if not finite:
        raise RuntimeError("non-finite BF16 result or gradient")
    return loss.item()
