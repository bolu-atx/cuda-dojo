# add_dojo_level(<name>
#   SOURCES <kernel .cu files...>   # compiled into a STATIC kernel library
#   DEMO    <demo main .cu>         # builds <name>_demo executable (optional)
#   TEST    <test .cu>              # builds <name>_test + registers CTest (optional)
# )
#
# Every level follows the same shape: kernels live in a small library that both
# the demo and the test link against, so the canonical host+kernel code is
# written once and exercised two ways.
function(add_dojo_level NAME)
  cmake_parse_arguments(L "" "" "SOURCES;DEMO;TEST" ${ARGN})

  add_library(${NAME} STATIC ${L_SOURCES})
  target_link_libraries(${NAME} PUBLIC dojo_common)
  target_include_directories(${NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  # Separable compilation lets kernels call __device__ functions across TUs.
  set_target_properties(${NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  if(L_DEMO)
    add_executable(${NAME}_demo ${L_DEMO})
    target_link_libraries(${NAME}_demo PRIVATE ${NAME})
  endif()

  if(L_TEST)
    add_executable(${NAME}_test ${L_TEST})
    target_link_libraries(${NAME}_test PRIVATE ${NAME} dojo_test)
    add_test(NAME ${NAME} COMMAND ${NAME}_test)
  endif()
endfunction()
