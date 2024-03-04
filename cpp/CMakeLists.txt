cmake_minimum_required(VERSION 3.2)

project(mlx-lite LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests for mlx-lite" ON)

add_library(mlx-lite)

include(FetchContent)

find_package(MLX CONFIG REQUIRED)
find_package(flatbuffers REQUIRED)

target_include_directories(
  mlx-lite
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(mlx-lite PUBLIC flatbuffers mlx)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mlx-lite)

if (BUILD_TESTS)
  include(CTest)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif()