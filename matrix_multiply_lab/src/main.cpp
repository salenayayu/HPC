#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include "matrix_multiply_cpu.h"
#include "matrix_multiply_cuda.h"

using namespace std;
using namespace chrono;

bool validateResult(const vector<double>& C_cpu, 
                    const vector<double>& C_gpu, 
                    int N, double epsilon = 1e-6) {
    for (int i = 0; i < N * N; ++i) {
        if (abs(C_cpu[i] - C_gpu[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void generateRandomMatrix(vector<double>& M, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int i = 0; i < N * N; ++i) {
        M[i] = dist(gen);
    }
}

int main() {
    vector<int> sizes = {100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000};
    
    ofstream results("../results/benchmark_results.csv");
    results << "Size,CPU_Time_ms,GPU_Time_ms,Speedup,Valid\n";
    
    cout << "============================================================" << endl;
    cout << "         MATRIX MULTIPLICATION: CPU vs GPU (CUDA)" << endl;
    cout << "============================================================" << endl;
    cout << setw(10) << "Size" << setw(15) << "CPU (ms)" << setw(15) << "GPU (ms)" 
         << setw(15) << "Speedup" << setw(15) << "Valid" << endl;
    cout << "------------------------------------------------------------" << endl;
    
    for (int N : sizes) {
        vector<double> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
        generateRandomMatrix(A, N);
        generateRandomMatrix(B, N);
        
        auto start = high_resolution_clock::now();
        multiplyMatricesCPU(A, B, C_cpu, N);
        auto cpuTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
        
        start = high_resolution_clock::now();
        multiplyMatricesGPU(A, B, C_gpu, N);
        auto gpuTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
        
        bool valid = validateResult(C_cpu, C_gpu, N);
        double speedup = (double)cpuTime / gpuTime;
        
        results << N << "," << cpuTime << "," << gpuTime << "," << speedup << "," << (valid ? "Yes" : "No") << "\n";
        
        cout << setw(10) << N << setw(15) << cpuTime << setw(15) << gpuTime 
             << setw(15) << fixed << setprecision(2) << speedup << "x"
             << setw(15) << (valid ? "✓" : "✗") << endl;
    }
    
    results.close();
    cout << "============================================================" << endl;
    return 0;
}
