add_dojo_level(level06
  SOURCES box_filter_shared.cu separable_blur.cu
  DEMO    demo.cu
  TEST    test_level06.cu
)

# Stretch profiling target: pad-vs-swizzle for bank-conflict-free shared memory.
# Self-contained (own kernels + main, no graded test), so it's a second executable
# rather than the level's single DEMO slot. Links level06 for dojo_common headers.
add_executable(level06_swizzle_demo swizzle_demo.cu)
target_link_libraries(level06_swizzle_demo PRIVATE level06)
