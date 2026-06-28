add_dojo_level(level05
  SOURCES box_filter_shared.cu separable_blur.cu
  DEMO    demo.cu
  TEST    test_level05.cu
)

# Stretch profiling target: pad-vs-swizzle for bank-conflict-free shared memory.
# Self-contained (own kernels + main, no graded test), so it's a second executable
# rather than the level's single DEMO slot. Links level05 for dojo_common headers.
add_executable(level05_swizzle_demo swizzle_demo.cu)
target_link_libraries(level05_swizzle_demo PRIVATE level05)
