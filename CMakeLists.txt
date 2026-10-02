cmake_minimum_required(VERSION 3.20)
project(vector_quantizations VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VQ_BUILD_TESTS "Build tests" ON)
option(VQ_BUILD_BENCHMARKS "Build benchmarks (requires Google Benchmark)" ON)

# Core library
add_library(vq src/vq.cpp src/pq.cpp)
add_library(vq::vq ALIAS vq)
target_include_directories(vq PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(vq PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>)

# Executable
add_executable(vq_cli apps/main.cpp)
target_link_libraries(vq_cli PRIVATE vq::vq)

# Small walkthrough example
add_executable(pq_example examples/main.cpp)
target_link_libraries(pq_example PRIVATE vq::vq)

# Tests
if(VQ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Benchmarks
if(VQ_BUILD_BENCHMARKS)
  find_package(benchmark REQUIRED)
  add_executable(bench_adc bench/bench_adc.cpp)
  target_link_libraries(bench_adc PRIVATE vq::vq benchmark::benchmark)
endif()
