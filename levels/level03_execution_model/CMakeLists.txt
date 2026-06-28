add_custom_target(level03)

add_executable(level03_demo demo.cpp)
target_link_libraries(level03_demo PRIVATE dojo_common)

add_executable(level03_test test_level03.cpp)
target_link_libraries(level03_test PRIVATE dojo_test)
add_test(NAME level03 COMMAND level03_test)
