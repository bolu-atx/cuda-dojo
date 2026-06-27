# CUDA Dojo — self-contained build environment.
#
# Encapsulates the entire CUDA toolkit (nvcc, cuBLAS, cuFFT, ...) and all build
# deps. The ONE thing it cannot contain is the NVIDIA kernel driver — that lives
# on the host and is bridged in at runtime by the NVIDIA Container Toolkit
# (`docker run --gpus all`). So:
#   * Compiling needs no GPU   -> set -DCMAKE_CUDA_ARCHITECTURES=all-major
#   * Running kernels needs a GPU host with the driver + container toolkit.
#
# Base image tag pins the toolkit version; -devel includes headers + libraries.
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

# Ubuntu 24.04 ships cmake 3.28 (>=3.24 ✓), gcc-13 (a valid CUDA 12.6 host
# compiler with good C++23 support), ninja, and python for the docs site.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make `--gpus all` expose everything (graphics+compute+utility) by default.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

# The repo is bind-mounted at run time (see the make docker-* targets), so we
# don't COPY sources in — this keeps the image a pure toolchain you can reuse
# across edits without rebuilding.
CMD ["bash"]
