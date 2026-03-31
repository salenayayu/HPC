#include "matrix_multiply_cpu.h"
#include <algorithm>

void multiplyMatricesCPU(const std::vector<double>& A,  const std::vector<double>& B, std::vector<double>& C, int N) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
}
