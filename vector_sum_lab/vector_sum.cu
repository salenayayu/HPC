#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;
using namespace chrono;

// ==================== CPU ====================
double vectorSumCPU(const vector<double>& vec) {
    double sum = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum;
}

// ==================== GPU (CUDA) ====================
// каждый поток суммирует несколько элементов
__global__ void vectorSumKernel(const double* input, double* partialSums, int N) {
    extern __shared__ double sharedData[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // загрузка shared memory
    if (globalIdx < N) {
        sharedData[tid] = input[globalIdx];
    } else {
        sharedData[tid] = 0.0;
    }
    __syncthreads();

    // редукция
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // частичная сумма блока
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

double vectorSumGPU(const vector<double>& vec) {
    int N = vec.size();
    size_t bytes = N * sizeof(double);

    // память на GPU
    double *d_input, *d_partialSums;
    cudaMalloc(&d_input, bytes);

    // Настройка блоков и сетки
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&d_partialSums, blocksPerGrid * sizeof(double));

    // Копируем данные на GPU
    cudaMemcpy(d_input, vec.data(), bytes, cudaMemcpyHostToDevice);

    // Запускаем ядро
    vectorSumKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_input, d_partialSums, N);

    // Копируем частичные суммы обратно
    vector<double> partialSums(blocksPerGrid);
    cudaMemcpy(partialSums.data(), d_partialSums, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // Финальная сумма на CPU
    double totalSum = 0.0;
    for (double s : partialSums) {
        totalSum += s;
    }

    // Очищаем память
    cudaFree(d_input);
    cudaFree(d_partialSums);

    return totalSum;
}

bool validateResult(double cpuSum, double gpuSum, double epsilon = 1e-6) {
    return abs(cpuSum - gpuSum) < epsilon;
}

// Генерация вектора
void generateRandomVector(vector<double>& vec, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-100.0, 100.0);

    for (int i = 0; i < N; ++i) {
        vec[i] = dist(gen);
    }
}

// временя выполнения
double benchmarkSum(double (*sumFunc)(const vector<double>&),
                    const vector<double>& vec) {
    // Прогрев
    sumFunc(vec);

    auto start = high_resolution_clock::now();
    double result = sumFunc(vec);
    auto end = high_resolution_clock::now();

    return duration_cast<microseconds>(end - start).count() / 1000.0; // в миллисекундах
}

int main() {
    vector<int> sizes = {1000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000};

    cout << setw(12) << "Vector Size"
         << setw(18) << "CPU (ms)"
         << setw(18) << "GPU (ms)"
         << setw(18) << "Speedup"
         << setw(15) << "Valid" << endl;
    cout << "------------------------------------------------------------------------------" << endl;

    for (int N : sizes) {
        cout << "Processing " << N << " elements... " << flush;

        vector<double> vec(N);
        generateRandomVector(vec, N);

        double cpuTime = benchmarkSum(vectorSumCPU, vec);
        double gpuTime = benchmarkSum(vectorSumGPU, vec);

        double cpuSum = vectorSumCPU(vec);
        double gpuSum = vectorSumGPU(vec);
        bool valid = validateResult(cpuSum, gpuSum);
        double speedup = cpuTime / gpuTime;

        cout << "\r" << setw(12) << N
             << setw(18) << fixed << setprecision(3) << cpuTime
             << setw(18) << setprecision(3) << gpuTime
             << setw(18) << setprecision(2) << speedup << "x"
             << setw(15) << (valid ? "OK" : "ERROR") << endl;
    }

    return 0;
}
