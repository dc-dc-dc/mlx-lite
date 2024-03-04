FetchContent_Declare(
  doctest
  GIT_REPOSITORY "https://github.com/onqtam/doctest"
  GIT_TAG "b7c21ec5ceeadb4951b00396fc1e4642dd347e5f"
)
FetchContent_MakeAvailable(doctest)

add_executable(tests ${CMAKE_CURRENT_LIST_DIR}/test.cpp)

target_sources(tests 
PRIVATE 
  ${CMAKE_CURRENT_LIST_DIR}/load_test.cpp
)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
target_link_libraries(tests PRIVATE mlx-lite doctest)
doctest_discover_tests(tests)
add_test(NAME tests COMMAND tests)