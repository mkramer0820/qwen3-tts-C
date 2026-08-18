#!/usr/bin/env bash
set -euo pipefail

action=${1:-setup}
compose=(docker compose --profile rocm)
docker_cmd=(docker)

# ROCm WSL device forwarding must use the engine in this Ubuntu distribution,
# not a Docker Desktop socket injected by WSL integration.
export DOCKER_HOST=unix:///var/run/docker.sock
export HSA_ENABLE_DXG_DETECTION=1
unset DOCKER_CONTEXT

step() {
    printf '\n==> %s\n' "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ $(uname -r) == *microsoft-standard-WSL2* ]] || die 'This setup must run inside an Ubuntu WSL2 distribution.'
source /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] || die 'Ubuntu 24.04 is required by the pinned ROCm container.'

install_host_rocm() {
    if [[ -e /dev/dxg && -r /opt/rocm/lib/librocdxg.so \
          && -r /opt/rocm/share/rocdxg/dids.conf ]]; then
        return
    fi

    [[ -e /dev/dxg ]] || die '/dev/dxg is missing. Install the current AMD Software: Adrenalin Edition Windows driver for WSL, run wsl --shutdown in PowerShell, and retry.'
    step 'Installing AMD ROCm 7.2.1 and ROCDXG 1.2.0 for WSL'
    sudo apt-get update
    sudo apt-get install -y ca-certificates wget
    wget -q https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb -O /tmp/amdgpu-install.deb
    sudo apt-get install -y /tmp/amdgpu-install.deb
    sudo amdgpu-install -y --usecase=rocm --no-dkms
    wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb -O /tmp/rocdxg-roct.deb
    sudo apt-get install -y /tmp/rocdxg-roct.deb

    [[ -r /opt/rocm/lib/librocdxg.so ]] || die 'ROCDXG installed without librocdxg.so. Review the package errors above.'
    [[ -r /opt/rocm/share/rocdxg/dids.conf ]] || die 'ROCDXG installed without dids.conf. Review the package errors above.'
}

install_docker() {
    if ! dpkg-query -W -f='${Status}' docker.io 2>/dev/null | grep -q 'install ok installed' \
       || ! docker compose version >/dev/null 2>&1; then
        step 'Installing Docker Engine and Compose inside Ubuntu WSL'
        sudo apt-get update
        sudo apt-get install -y docker.io docker-compose-v2
    fi

    if ! docker info >/dev/null 2>&1; then
        step 'Starting Docker Engine'
        sudo systemctl enable --now docker 2>/dev/null || sudo service docker start
    fi

    if ! docker info >/dev/null 2>&1; then
        if sudo docker info >/dev/null 2>&1; then
            compose=(sudo docker compose --profile rocm)
            docker_cmd=(sudo docker)
            printf 'NOTE: using sudo for Docker. Log out of Ubuntu and back in after adding your user to the docker group.\n'
            sudo usermod -aG docker "$USER"
        else
            die 'Docker Engine is installed but is not running.'
        fi
    fi

    local server_os
    server_os=$("${docker_cmd[@]}" info --format '{{.OperatingSystem}}' 2>/dev/null || true)
    if [[ $server_os == *Docker\ Desktop* ]]; then
        die 'Docker is connected to Docker Desktop, not Ubuntu Docker Engine. Disable Docker Desktop WSL integration for this distro, run wsl --shutdown in PowerShell, and retry.'
    fi
}

run_doctor() {
    install_host_rocm
    install_docker
    step 'Validating the Compose configuration'
    "${compose[@]}" config --quiet
    step 'Checking RDNA GPU access inside the container'
    "${compose[@]}" run --rm --build rocm-wsl doctor
}

case "$action" in
    setup)
        install_host_rocm
        install_docker
        step 'Building the ROCm development image'
        "${compose[@]}" build rocm-wsl
        run_doctor
        printf '\nREADY: run .\\rocm-wsl.ps1 -Action build from PowerShell to compile and self-test C inference.\n'
        ;;
    doctor)
        run_doctor
        ;;
    build)
        run_doctor
        step 'Building and self-testing C inference for the detected GPU target'
        "${compose[@]}" run --rm rocm-wsl build
        ;;
    shell)
        install_host_rocm
        install_docker
        "${compose[@]}" run --rm --build rocm-wsl shell
        ;;
    *)
        die "Unknown action: $action"
        ;;
esac
