cmake_minimum_required(VERSION 3.10)
project(MatrixMultiply)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(matrix_multiply 
    src/main.cpp
    src/matrix_multiply_cpu.cpp
    src/matrix_multiply_cuda.cu
)

target_include_directories(matrix_multiply PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(matrix_multiply ${CUDA_LIBRARIES})
