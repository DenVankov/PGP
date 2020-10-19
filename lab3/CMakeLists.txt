cmake_minimum_required(VERSION 3.17)
project(untitled2 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(untitled2 lab3.cu)

set_target_properties(
        untitled2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)