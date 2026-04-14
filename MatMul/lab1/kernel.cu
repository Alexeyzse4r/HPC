#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iomanip>

const int NUM_RUNS = 30;
const int BLOCK_DIM = 16;

// Произведение матриц на СPU реализация через 3 вложенных цикла
void cpu_matmul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}


// Произведение матриц на GPU
__global__ void gpu_matmul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[i * N + k] * B[k * N + j];
        C[i * N + j] = sum;
    }
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<int> sizes = { 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000 };

    for (int N : sizes) {
        std::vector<float> A(N * N, 0.0f), B(N * N, 0.0f), C_cpu(N * N, 0.0f), C_gpu(N * N, 0.0f);

        for (size_t i = 0; i < N * N; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }

        double cpu_all = 0, gpu_all = 0, gpu_core = 0;
        float max_err = 0.0f;

        for (int run = 0; run < NUM_RUNS; ++run) {

            //  CPU замер времени
            auto cpu_st = std::chrono::high_resolution_clock::now();
            cpu_matmul(A.data(), B.data(), C_cpu.data(), N);
            cpu_all += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cpu_st).count();

            float* d_A, * d_B, * d_C;
            size_t sz = N * N * sizeof(float);
            cudaMalloc(&d_A, sz); cudaMalloc(&d_B, sz); cudaMalloc(&d_C, sz);

            // перенос с проца на девайс и начало первого замера для GPU
            auto gpu_st1 = std::chrono::high_resolution_clock::now();
            cudaMemcpy(d_A, A.data(), sz, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B.data(), sz, cudaMemcpyHostToDevice);

            // второй замер времени для GPU
            auto gpu_st2 = std::chrono::high_resolution_clock::now();
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);
            gpu_matmul<<<blocks, threads>>> (d_A, d_B, d_C, N);
            cudaDeviceSynchronize();
            gpu_core += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - gpu_st2).count();

            // перенос с девайса на хост
            cudaMemcpy(C_gpu.data(), d_C, sz, cudaMemcpyDeviceToHost);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

            gpu_all += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - gpu_st1).count();

            // Ищем максимальную ошибку
            for (int i = 0; i < N * N; ++i)
                max_err = fmaxf(max_err, fabsf(C_cpu[i] - C_gpu[i]));
        }

        // Берем среднее из полученного потому что есть NUM_RUNS
        cpu_all /= NUM_RUNS;
        gpu_all /= NUM_RUNS;
        gpu_core /= NUM_RUNS;

        std::cout << N << " | " << cpu_all << " | " << gpu_all << " | " << gpu_core << " | " << cpu_all / gpu_all << " | " << cpu_all / gpu_core << " | " << max_err << "\n";
    }

    return 0;
}