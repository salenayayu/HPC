#ifndef MATRIX_MULTIPLY_CUDA_H
#define MATRIX_MULTIPLY_CUDA_H

#include <vector>

void multiplyMatricesGPU(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N);

#endif
