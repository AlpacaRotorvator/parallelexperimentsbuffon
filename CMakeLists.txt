cmake_minimum_required(VERSION 2.8)
project(buffoncuda LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)

include(CTest)

add_library( misc STATIC
  misc.hu
  misc.cu)

set_target_properties( misc
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

add_executable(
  buffoncuda
  main.cu)

set_target_properties(buffoncuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(buffoncuda PRIVATE misc)
