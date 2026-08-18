#!/usr/bin/env bash
set -euo pipefail

action=${1:-setup}
compose=(docker compose --profile rocm)
docker_cmd=(docker)
rocm_version=7.2.1
rocdxg_version=1.2.0
amdgpu_installer_version=7.2.1.70201-1
amdgpu_installer_sha256=4c0338a241c15b12c14eb3aeb4012ea0d55dba681737ea8482248041a16c2afa
rocdxg_sha256=3ed9526719290cd8f590150dad8ea0f234fa779bea6a4c9a8449d7ae6b8cfb6e

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

package_version() {
    dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true
}

version_matches() {
    local actual=$1 expected=$2
    [[ $actual == "$expected" || $actual == "$expected".* || $actual == "$expected"-* ]]
}

download_verified() {
    local url=$1 output=$2 expected_sha256=$3
    wget -q "$url" -O "$output"
    printf '%s  %s\n' "$expected_sha256" "$output" | sha256sum --check --status \
        || die "SHA-256 verification failed for $url"
}

verify_host_rocm() {
    local installed_rocm installed_rocdxg
    installed_rocm=$(package_version rocm-core)
    installed_rocdxg=$(package_version rocdxg-roct)
    version_matches "$installed_rocm" "$rocm_version" \
        || die "Expected rocm-core $rocm_version, found '${installed_rocm:-not installed}'. Remove the mismatched ROCm packages or install the documented version."
    version_matches "$installed_rocdxg" "$rocdxg_version" \
        || die "Expected rocdxg-roct $rocdxg_version, found '${installed_rocdxg:-not installed}'. ROCm 7.2.x maps to ROCDXG 1.2.0 in AMD's compatibility table."
    [[ -r /opt/rocm/lib/librocdxg.so ]] || die 'ROCDXG is installed without /opt/rocm/lib/librocdxg.so.'
    [[ -r /opt/rocm/share/rocdxg/dids.conf ]] || die 'ROCDXG is installed without dids.conf.'
    printf 'Host ROCm packages: rocm-core=%s, rocdxg-roct=%s\n' "$installed_rocm" "$installed_rocdxg"
}

[[ $(uname -r) == *microsoft-standard-WSL2* ]] || die 'This setup must run inside an Ubuntu WSL2 distribution.'
source /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 ]] || die 'Ubuntu 24.04 is required by the pinned ROCm container.'

install_host_rocm() {
    local installed_rocm installed_rocdxg
    installed_rocm=$(package_version rocm-core)
    installed_rocdxg=$(package_version rocdxg-roct)
    if [[ -n $installed_rocm && -n $installed_rocdxg ]]; then
        verify_host_rocm
        return
    fi

    if [[ -n $installed_rocm ]] && ! version_matches "$installed_rocm" "$rocm_version"; then
        die "Expected rocm-core $rocm_version, found '$installed_rocm'. Remove the mismatched ROCm packages before continuing."
    fi
    if [[ -n $installed_rocdxg ]] && ! version_matches "$installed_rocdxg" "$rocdxg_version"; then
        die "Expected rocdxg-roct $rocdxg_version, found '$installed_rocdxg'. Remove the mismatched ROCDXG package before continuing."
    fi

    [[ -e /dev/dxg ]] || die '/dev/dxg is missing. Install the current AMD Software: Adrenalin Edition Windows driver for WSL, run wsl --shutdown in PowerShell, and retry.'
    step "Installing AMD ROCm $rocm_version and ROCDXG $rocdxg_version for WSL"
    sudo apt-get update
    sudo apt-get install -y ca-certificates wget
    if [[ -z $installed_rocm ]]; then
        download_verified \
            "https://repo.radeon.com/amdgpu-install/$rocm_version/ubuntu/noble/amdgpu-install_${amdgpu_installer_version}_all.deb" \
            /tmp/amdgpu-install.deb "$amdgpu_installer_sha256"
        sudo apt-get install -y /tmp/amdgpu-install.deb
        sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
    fi
    if [[ -z $installed_rocdxg ]]; then
        download_verified \
            "https://github.com/ROCm/librocdxg/releases/download/v$rocdxg_version/rocdxg-roct_${rocdxg_version}_amd64.deb" \
            /tmp/rocdxg-roct.deb "$rocdxg_sha256"
        sudo apt-get install -y /tmp/rocdxg-roct.deb
    fi
    verify_host_rocm
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
