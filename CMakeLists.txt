cmake_minimum_required(VERSION 3.25)
project(lab03 CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc")
add_executable(global ex03/VectorAdd.cu)
add_executable(shared_static ex03/ShareStaticVectorAdd.cu)
add_executable(shared_dynamic ex03/ShareDynamicVectorAdd.cu)

set_target_properties(global PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(shared_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(shared_dynamic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
