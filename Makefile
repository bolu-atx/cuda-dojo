# CUDA Dojo — convenience wrapper around CMake/CTest + the docs site.
#
# Per-level targets are generated automatically from the directories under
# levels/, so new levels light up here the moment they're added. For a level
# directory named `levelNN_topic`, the CMake target is `levelNN` (the first
# argument to add_dojo_level), which is what these targets drive.
#
#   make                 configure + build everything
#   make level01         build level 1's kernel lib + demo
#   make level01-test    build + run level 1's tests
#   make test            run every level's tests
#   make dep-check       verify the toolchain (nvcc, cmake, generator, docs)
#   make docs            build the static docs site into ./out/docs
#   make docs-serve      live-preview the docs at http://127.0.0.1:9090
#   make clean           remove the CMake build directory
#   make distclean       also remove ./out and ./.venv
#   make help            list all targets (including per-level ones)

BUILD_DIR := build
VENV      := .venv
JOBS      ?= $(shell (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4))

# Prefer Ninja when present; otherwise let CMake pick its default generator.
ifneq ($(shell command -v ninja 2>/dev/null),)
GEN_FLAG := -G Ninja
else
GEN_FLAG :=
endif

# Discover levels: levels/level07_libraries -> level07
LEVEL_DIRS  := $(sort $(wildcard levels/level*))
LEVELS      := $(foreach d,$(LEVEL_DIRS),$(firstword $(subst _, ,$(notdir $(d)))))
LEVEL_TESTS := $(addsuffix -test,$(LEVELS))

.DEFAULT_GOAL := help
IMAGE       := cuda-dojo:dev
DOCKER_DIR  := build-docker

.PHONY: all configure build test clean distclean docs docs-serve docs-deps \
        dep-check help docker-build docker-compile docker-test docker-shell \
        $(LEVELS) $(LEVEL_TESTS)

# ---- build ---------------------------------------------------------------
all: build

# Idempotent configure. Re-runs are cheap; CMake no-ops if nothing changed.
configure:
	@cmake -B $(BUILD_DIR) $(GEN_FLAG)

build: configure
	@cmake --build $(BUILD_DIR) -j $(JOBS)

# Per-level build: kernel library + demo executable for that level.
$(LEVELS): %: configure
	@cmake --build $(BUILD_DIR) -j $(JOBS) --target $* --target $*_demo

# Per-level test: build that level's test exe, then run just its CTest entry.
$(LEVEL_TESTS): %-test: configure
	@cmake --build $(BUILD_DIR) -j $(JOBS) --target $*_test
	@ctest --test-dir $(BUILD_DIR) -R "^$*$$" --output-on-failure

# ---- test ----------------------------------------------------------------
test: build
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

# ---- docs ----------------------------------------------------------------
docs-deps:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(VENV)/bin/pip install -q -r requirements.txt

docs: docs-deps
	@$(VENV)/bin/mkdocs build

docs-serve: docs-deps
	@$(VENV)/bin/mkdocs serve -a 127.0.0.1:9090

# ---- docker --------------------------------------------------------------
# The image holds the whole CUDA toolkit + build deps; the repo is bind-mounted
# so edits don't require a rebuild. A separate build dir (build-docker/) keeps
# container artifacts from clobbering a host build/.
#
#   make docker-build     build the toolchain image
#   make docker-compile   compile in-container WITHOUT a GPU (verifies it builds)
#   make docker-test      build + run tests in-container (needs --gpus all)
#   make docker-shell     interactive shell in-container (--gpus all)
DOCKER_RUN  := docker run --rm -v "$(CURDIR)":/workspace -w /workspace
DOCKER_GPU  := --gpus all

docker-build:
	@docker build -t $(IMAGE) .

docker-compile: docker-build
	@$(DOCKER_RUN) $(IMAGE) bash -c '\
	  cmake -B $(DOCKER_DIR) -G Ninja -DCMAKE_CUDA_ARCHITECTURES=all-major && \
	  cmake --build $(DOCKER_DIR) -j $(JOBS)'

docker-test: docker-build
	@$(DOCKER_RUN) $(DOCKER_GPU) $(IMAGE) bash -c '\
	  cmake -B $(DOCKER_DIR) -G Ninja -DCMAKE_CUDA_ARCHITECTURES=native && \
	  cmake --build $(DOCKER_DIR) -j $(JOBS) && \
	  ctest --test-dir $(DOCKER_DIR) --output-on-failure'

docker-shell: docker-build
	@$(DOCKER_RUN) $(DOCKER_GPU) -it $(IMAGE) bash

# ---- housekeeping --------------------------------------------------------
clean:
	@rm -rf $(BUILD_DIR) $(DOCKER_DIR)
	@echo "removed $(BUILD_DIR)/ and $(DOCKER_DIR)/"

distclean: clean
	@rm -rf out $(VENV)
	@echo "removed out/ and $(VENV)/"

# ---- dependency check ----------------------------------------------------
# Non-fatal report of what's installed. Exits non-zero only if a required
# build tool (nvcc, cmake) is missing, so it's usable in CI gating.
dep-check:
	@echo "CUDA Dojo dependency check"
	@echo "=========================="
	@ok=1; \
	check() { \
	  if command -v "$$1" >/dev/null 2>&1; then \
	    printf "  [ok]   %-10s %s\n" "$$1" "$$($$2 2>&1 | head -n1)"; \
	  else \
	    printf "  [MISS] %-10s %s\n" "$$1" "$$3"; \
	    [ "$$4" = "req" ] && ok=0; \
	  fi; }; \
	echo "build (required):"; \
	check nvcc  "nvcc --version | tail -n2 | head -n1" "install the CUDA Toolkit" req; \
	check cmake "cmake --version" "brew/apt install cmake (>=3.24)" req; \
	echo "build (optional):"; \
	check ninja "ninja --version" "faster builds: apt/brew install ninja" opt; \
	echo "profiling (optional, Level 5+):"; \
	check nsys "nsys --version | head -n1" "Nsight Systems (ships with CUDA Toolkit)" opt; \
	check ncu  "ncu --version | head -n1"  "Nsight Compute (ships with CUDA Toolkit)" opt; \
	check compute-sanitizer "compute-sanitizer --version | head -n1" "memcheck (ships with CUDA Toolkit)" opt; \
	echo "docs (optional):"; \
	check python3 "python3 --version" "needed for the mkdocs docs site" opt; \
	echo; \
	if [ "$$ok" = "1" ]; then echo "Required build tools present."; \
	else echo "Missing a required build tool — see [MISS] above."; exit 1; fi

# ---- help ----------------------------------------------------------------
help:
	@echo "CUDA Dojo make targets   (default: help)"
	@echo "  all / build     configure + build everything"
	@echo "  test            run all CTest tests"
	@echo "  dep-check       verify toolchain (nvcc, cmake, generator, profilers)"
	@echo "  docs            build static docs site into ./out/docs"
	@echo "  docs-serve      live docs preview (http://127.0.0.1:9090)"
	@echo "  docker-build    build the CUDA toolchain image"
	@echo "  docker-compile  compile in-container, no GPU needed (all-major)"
	@echo "  docker-test     build + test in-container (needs --gpus all)"
	@echo "  docker-shell    interactive shell in-container (--gpus all)"
	@echo "  clean           remove ./$(BUILD_DIR) and ./$(DOCKER_DIR)"
	@echo "  distclean       also remove ./out and ./$(VENV)"
	@echo
	@echo "per-level (auto-discovered from levels/):"
	@for l in $(LEVELS); do printf "  %-15s build %s\n" "$$l" "$$l"; done
	@echo "  <level>-test    build + run that level's tests, e.g. 'make level01-test'"
