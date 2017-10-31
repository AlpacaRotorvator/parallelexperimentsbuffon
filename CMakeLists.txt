cmake_minimum_required(VERSION 2.8)

project(buffoncuda LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)
include(CTest)

#set (CMAKE_EXPORT_COMPILE_COMMANDS "ON")

include_directories(/opt/cuda/include /opt/cuda/samples/common/inc)

#list (APPEND CMAKE_CUDA_FLAGS "-lineinfo -Wno-deprecated-gpu-targets")

add_library( cutils STATIC
  cutils.hu
  cutils.cu)
set_target_properties( cutils
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       )
		     
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

add_library( batchrng_kernel STATIC
  batchrng_kernel.hu
  batchrng_kernel.cu)
target_compile_features(batchrng_kernel PUBLIC cxx_std_11)
set_target_properties( batchrng_kernel
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       )	
		     

add_executable(
  buffoncuda
  main.cu)

set_target_properties(buffoncuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(buffoncuda PUBLIC cxx_std_11)
target_link_libraries(buffoncuda PRIVATE naive_kernel)
target_link_libraries(buffoncuda PRIVATE ${CUDA_curand_LIBRARY})
target_link_libraries(buffoncuda PRIVATE batchrng_kernel)
target_link_libraries(buffoncuda PRIVATE misc)
target_link_libraries(buffoncuda PRIVATE cutils)
