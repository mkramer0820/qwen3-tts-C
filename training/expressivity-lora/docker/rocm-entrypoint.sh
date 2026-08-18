#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/qwen3-tts-C

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

gpu_arch() {
    python3 -c 'import torch; print(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0])'
}

select_gpu() {
    local count index name
    count=$(python3 -c 'import torch; print(torch.cuda.device_count())')
    [[ $count =~ ^[1-9][0-9]*$ ]] || die 'ROCm PyTorch does not see a GPU through ROCDXG.'
    index=$(python3 -c 'import torch; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; print(next((i for i,n in enumerate(names) if "Radeon AI PRO R9700" in n), 0))')
    export HIP_VISIBLE_DEVICES=$index
    name=$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')
    printf 'Selected ROCm GPU: %s (host HIP index %s)\n' "$name" "$index"
}

doctor() {
    [[ -e /dev/dxg ]] || die '/dev/dxg is missing. Run this container from Ubuntu WSL2 with the AMD Windows driver installed.'
    [[ -r /usr/lib/libdxcore.so ]] || die 'The WSL libdxcore mount is missing.'
    [[ -r /usr/lib/librocdxg.so ]] || die 'The WSL ROCDXG runtime mount is missing.'
    [[ -r /usr/share/rocdxg/dids.conf ]] || die 'The WSL ROCDXG device-ID configuration mount is missing.'
    [[ ${HSA_ENABLE_DXG_DETECTION:-} == 1 ]] || die 'HSA_ENABLE_DXG_DETECTION=1 is required for ROCm 7.2.x ROCDXG.'
    command -v rocminfo >/dev/null || die 'rocminfo is not installed in the image.'
    select_gpu

    local arch
    arch=$(gpu_arch)
    [[ -n "$arch" ]] || die 'rocminfo did not report a GPU agent.'
    printf 'ROCm GPU architecture: %s\n' "$arch"
    case "$arch" in
        gfx11*|gfx12*) ;;
        *) die "This container workflow is scoped to RDNA3/RDNA4 (gfx11*/gfx12*), not $arch." ;;
    esac
    if [[ "$arch" != gfx1201 ]]; then
        printf 'NOTE: AMD Radeon AI PRO R9700 acceptance uses gfx1201; detected %s.\n' "$arch" >&2
    fi

    python3 "$repo/training/expressivity-lora/check_rocm.py"
}

build_inference() {
    doctor
    local arch
    arch=$(gpu_arch)
    make -C "$repo" rocm ROCM_PATH=/opt/rocm ROCM_ARCH="$arch"
    "$repo/qwen_tts" --gpu-selftest --backend rocm
}

case "${1:-help}" in
    doctor)
        doctor
        ;;
    build)
        build_inference
        ;;
    infer)
        shift
        doctor
        if [[ ! -x "$repo/qwen_tts" ]] || ! "$repo/qwen_tts" --gpu-selftest --backend rocm; then
            build_inference
        fi
        exec "$repo/qwen_tts" "$@" --backend rocm
        ;;
    train)
        shift
        doctor
        exec python3 "$repo/training/expressivity-lora/train_lora.py" --require-rocm "$@"
        ;;
    shell)
        exec bash
        ;;
    help|-h|--help)
        cat <<'EOF'
qwen-rocm commands:
  doctor             Verify WSL mounts, ROCm, PyTorch, gfx target, and BF16
  build              Compile C inference for the detected GPU and self-test it
  infer <args...>    Build if needed, then run qwen_tts --backend rocm
  train <args...>    Run train_lora.py with --require-rocm
  shell              Open an interactive shell
EOF
        ;;
    *)
        exec "$@"
        ;;
esac
