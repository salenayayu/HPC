#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;
using namespace chrono;

#define BLOCK_SIZE 16
#define RADIUS 1

// Ядро билатерального фильтра
__global__ void bilateralFilterKernel(const unsigned char* input, unsigned char* output, 
                                       int width, int height, float sigma_d, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float center = (float)input[idx];
    
    float sum_w = 0.0f;
    float sum_val = 0.0f;
    
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx < 0) nx = 0;
            if (nx >= width) nx = width - 1;
            if (ny < 0) ny = 0;
            if (ny >= height) ny = height - 1;
            
            float neighbor = (float)input[ny * width + nx];
            
            // пространственный вес
            float spatial = expf(-(dx*dx + dy*dy) / (2.0f * sigma_d * sigma_d));
            
            // диапазонный вес
            float diff = neighbor - center;
            float range = expf(-(diff * diff) / (2.0f * sigma_r * sigma_r));
            
            float w = spatial * range;
            sum_w += w;
            sum_val += w * neighbor;
        }
    }
    
    if (sum_w > 0.0001f) {
        output[idx] = (unsigned char)(sum_val / sum_w);
    } else {
        output[idx] = input[idx];
    }
}

// CPU версия
void bilateralFilterCPU(const unsigned char* input, unsigned char* output, 
                        int width, int height, float sigma_d, float sigma_r) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float center = (float)input[idx];
            
            float sum_w = 0.0f;
            float sum_val = 0.0f;
            
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx < 0) nx = 0;
                    if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0;
                    if (ny >= height) ny = height - 1;
                    
                    float neighbor = (float)input[ny * width + nx];
                    float spatial = expf(-(dx*dx + dy*dy) / (2.0f * sigma_d * sigma_d));
                    float diff = neighbor - center;
                    float range = expf(-(diff * diff) / (2.0f * sigma_r * sigma_r));
                    float w = spatial * range;
                    
                    sum_w += w;
                    sum_val += w * neighbor;
                }
            }
            
            if (sum_w > 0.0001f) {
                output[idx] = (unsigned char)(sum_val / sum_w);
            } else {
                output[idx] = input[idx];
            }
        }
    }
}

// BMP файл
bool saveBMP(const string& filename, unsigned char* image, int width, int height) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) return false;
    
    int rowSize = (width * 3 + 3) & ~3;
    int imageSize = rowSize * height;
    int fileSize = 54 + imageSize;
    
    unsigned char header[54] = {
        0x42, 0x4D, (unsigned char)(fileSize), (unsigned char)(fileSize >> 8), 
        (unsigned char)(fileSize >> 16), (unsigned char)(fileSize >> 24), 0, 0, 0, 0, 54, 0, 0, 0,
        40, 0, 0, 0, (unsigned char)width, (unsigned char)(width >> 8), 
        (unsigned char)(width >> 16), (unsigned char)(width >> 24),
        (unsigned char)height, (unsigned char)(height >> 8), 
        (unsigned char)(height >> 16), (unsigned char)(height >> 24),
        1, 0, 24, 0, 0, 0, 0, 0, (unsigned char)(imageSize), 
        (unsigned char)(imageSize >> 8), (unsigned char)(imageSize >> 16), 
        (unsigned char)(imageSize >> 24), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    
    file.write((char*)header, 54);
    
    vector<unsigned char> row(rowSize, 0);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            unsigned char val = image[y * width + x];
            row[x * 3] = val;
            row[x * 3 + 1] = val;
            row[x * 3 + 2] = val;
        }
        file.write((char*)row.data(), rowSize);
    }
    
    file.close();
    return true;
}

int main() {
    int width = 256, height = 256;
    
    // параметры фильтра
    float sigma_d = 1.0f;
    float sigma_r = 30.0f;
    
    cout << "============================================================" << endl;
    cout << "         БИЛАТЕРАЛЬНЫЙ ФИЛЬТР: CPU vs GPU (CUDA)" << endl;
    cout << "============================================================" << endl;
    cout << "Размер: " << width << " x " << height << endl;
    cout << "sigma_d = " << sigma_d << ", sigma_r = " << sigma_r << endl;
    cout << "------------------------------------------------------------" << endl;
    
    ifstream file("test_input.bmp", ios::binary);
    
    file.seekg(54);
    int img_size = width * height;
    unsigned char* h_input = new unsigned char[img_size];
    file.read((char*)h_input, img_size);
    file.close();
    
    unsigned char* h_output_cpu = new unsigned char[img_size];
    unsigned char* h_output_gpu = new unsigned char[img_size];
    
    // ==================== CPU ====================
    cout << "\n[CPU] Запуск..." << endl;
    auto start = high_resolution_clock::now();
    bilateralFilterCPU(h_input, h_output_cpu, width, height, sigma_d, sigma_r);
    auto cpu_time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    cout << "[CPU] Время: " << cpu_time << " ms" << endl;
    saveBMP("test_cpu.bmp", h_output_cpu, width, height);
    
    // ==================== GPU ====================
    cout << "\n[GPU] Запуск..." << endl;
    
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);
    
    start = high_resolution_clock::now();
    bilateralFilterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, sigma_d, sigma_r);
    cudaDeviceSynchronize();
    auto gpu_time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    
    cudaMemcpy(h_output_gpu, d_output, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    
    cout << "[GPU] Время: " << gpu_time << " ms" << endl;
    saveBMP("test_gpu.bmp", h_output_gpu, width, height);
    
    // ==================== РЕЗУЛЬТАТЫ ====================
    cout << "\n============================================================" << endl;
    cout << "РЕЗУЛЬТАТЫ" << endl;
    cout << "============================================================" << endl;
    cout << "CPU время: " << cpu_time << " ms" << endl;
    cout << "GPU время: " << gpu_time << " ms" << endl;
    cout << "Ускорение: " << fixed << setprecision(2) << (float)cpu_time / gpu_time << "x" << endl;
    
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    
    return 0;
}
