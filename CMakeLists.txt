cmake_minimum_required(VERSION 2.8)
project(buffoncuda LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)

include(CTest)

set (DCMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-5")
set (CMAKE_EXPORT_COMPILE_COMMANDS "ON")

add_library( misc STATIC
  misc.hu
  misc.cu)
set_target_properties( misc
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       )

add_library( naive_kernel STATIC
  naive_kernel.hu
  naive_kernel.cu)
target_compile_features(naive_kernel PUBLIC cxx_std_11)
set_target_properties( naive_kernel
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       )		     
		     

add_executable(
  buffoncuda
  main.cu)

set_target_properties(buffoncuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(buffoncuda PRIVATE naive_kernel)
target_link_libraries(buffoncuda PRIVATE misc)
