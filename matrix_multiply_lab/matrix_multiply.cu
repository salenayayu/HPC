#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;
using namespace chrono;

// ==================== CPU  ====================
void multiplyMatricesCPU(const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
}

// ==================== GPU ====================
__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void multiplyMatricesGPU(const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    double *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(double);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


bool validateResult(const vector<double>& C_cpu, const vector<double>& C_gpu, int N, double epsilon = 1e-6) {
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

    system("mkdir -p results");
    
    ofstream results("results/benchmark_results.csv");
    results << "Size,CPU_Time_ms,GPU_Time_ms,Speedup,Valid\n";
    
    cout << "============================================================" << endl;
    cout << "         MATRIX MULTIPLICATION: CPU vs GPU (CUDA)" << endl;
    cout << "============================================================" << endl;
    cout << setw(10) << "Size" 
         << setw(18) << "CPU (ms)" 
         << setw(18) << "GPU (ms)" 
         << setw(18) << "Speedup" 
         << setw(15) << "Valid" << endl;
    cout << "------------------------------------------------------------" << endl;
    
    for (int N : sizes) {
        cout << "Processing " << N << "x" << N << "... " << flush;
        
        vector<double> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
        
        generateRandomMatrix(A, N);
        generateRandomMatrix(B, N);
        
        // CPU Benchmark
        auto start = high_resolution_clock::now();
        multiplyMatricesCPU(A, B, C_cpu, N);
        auto cpuTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
        
        // GPU Benchmark
        start = high_resolution_clock::now();
        multiplyMatricesGPU(A, B, C_gpu, N);
        auto gpuTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
        
        bool valid = validateResult(C_cpu, C_gpu, N);
        double speedup = (double)cpuTime / gpuTime;
        
        results << N << "," << cpuTime << "," << gpuTime << "," << speedup << "," << (valid ? "Yes" : "No") << "\n";
        
        cout << "\r" << setw(10) << N 
             << setw(18) << fixed << setprecision(0) << cpuTime 
             << setw(18) << setprecision(0) << gpuTime 
             << setw(18) << setprecision(2) << speedup << "x"
             << setw(15) << (valid ? "✓ OK" : "✗ ERROR") << endl;
    }
    
    results.close();
    cout << "============================================================" << endl;
    cout << "Benchmark completed! Results saved to results/benchmark_results.csv" << endl;
    
    return 0;
}
