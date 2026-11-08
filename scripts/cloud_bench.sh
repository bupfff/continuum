#!/usr/bin/env bash
# Builds and runs Continuum's benchmark (experiments/room3d/src/bin/bench.rs)
# on a Linux machine with an NVIDIA GPU. See docs/BENCHMARK.md.
#
# Run from the repository's root on the GPU machine:
#   bash scripts/cloud_bench.sh            # the full suite (about 15 minutes)
#   bash scripts/cloud_bench.sh --quick    # one flight of each kind
#
# Assumes Ubuntu 22.04 or 24.04 with the NVIDIA driver installed (cloud
# "GPU" or "deep learning" images have it). In a container, start it with
# NVIDIA_DRIVER_CAPABILITIES=all: Vulkan needs the driver's graphics part.
set -euo pipefail

echo "== GPU"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found: install the NVIDIA driver first (see docs/BENCHMARK.md)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential pkg-config git curl libvulkan1 vulkan-tools >/dev/null

echo "== Vulkan"
if ! vulkaninfo --summary 2>/dev/null | grep -qi nvidia; then
  # Server images often ship the driver without its Vulkan part.
  major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  echo "No NVIDIA Vulkan device; installing libnvidia-gl-${major}."
  sudo apt-get install -y -qq "libnvidia-gl-${major}" >/dev/null \
    || sudo apt-get install -y -qq "libnvidia-gl-${major}-server" >/dev/null \
    || { echo "Could not install the driver's Vulkan library; see docs/BENCHMARK.md." >&2; exit 1; }
fi
vulkaninfo --summary 2>/dev/null | grep -iE 'deviceName|driverVersion|apiVersion' | head -6
if ! vulkaninfo 2>/dev/null | grep -q VK_KHR_ray_query; then
  echo "Note: no VK_KHR_ray_query on this GPU; camera rays will use the compute path."
fi

echo "== Rust"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustup update stable >/dev/null
cargo --version

echo "== Build"
cargo build --release -p room3d --bin bench

echo "== Benchmark"
WGPU_BACKEND=vulkan ./target/release/bench "$@"
echo "== Report: $(ls -t experiments/room3d/out/bench_*.txt | head -1)"
