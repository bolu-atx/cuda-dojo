cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# CUDA Dojo — a level-by-level CUDA learning project.
#
# Host code is C++23. Device (CUDA) code is C++20, which is the highest the
# CUDA toolchain reliably supports for __device__ code today. Each "level"
# lives under levels/ and registers a kernel library, a demo executable, and
# a CTest test via the add_dojo_level() helper in cmake/Dojo.cmake.
# ---------------------------------------------------------------------------

project(cuda_dojo LANGUAGES CXX)

# Fail early and clearly on machines without a CUDA compiler (e.g. macOS).
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "No CUDA compiler (nvcc) found.\n"
    "This project must be configured on a machine with the CUDA Toolkit "
    "installed and an NVIDIA GPU. macOS/Apple Silicon is not supported by "
    "CUDA — build on a Linux/Windows box with an NVIDIA GPU.")
endif()
enable_language(CUDA)

# C++ standards: host C++23, device C++20.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU(s) actually present in the build machine unless overridden,
# e.g. -DCMAKE_CUDA_ARCHITECTURES=80;86;90 for a fixed set.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Emit line info in Release so Nsight Compute can map SASS back to source —
# you'll want this from Level 5 onward.
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(Dojo)

# ---- shared interface targets -------------------------------------------
# Header-only common utilities (error checking, GPU timing, device info).
add_library(dojo_common INTERFACE)
target_include_directories(dojo_common INTERFACE "${CMAKE_SOURCE_DIR}/common")

# Header-only micro test harness.
add_library(dojo_test INTERFACE)
target_include_directories(dojo_test INTERFACE "${CMAKE_SOURCE_DIR}/common")

enable_testing()

add_subdirectory(levels/level01_vector_add)
add_subdirectory(levels/level02_thread_mapping)
add_subdirectory(levels/level03_execution_model)
add_subdirectory(levels/level04_memory_hierarchy)
add_subdirectory(levels/level05_performance)
add_subdirectory(levels/level06_shared_memory)
add_subdirectory(levels/level07_warps)
