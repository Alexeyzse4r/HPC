#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iomanip>

const int NUM_RUNS = 30;
//Заменил перебором
//const int BLOCK_DIM = 256;
//const int BLOCK_DIM = 128;

void cpu_sum(const float* vec, int n, double* result) {
    *result = 0.0;
    for (int i = 0; i < n; ++i)
        *result += vec[i];
}

__global__ void gpu_sum(const float* data, float* partial_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<int> sizes = { 1000, 10000, 100000, 1000000, 10000000, 100000000 };
    //решил поэкспериментировать какой оптимальный размер блока
    //std::vector<int> block_size = { 16, 32, 64, 128, 256, 512, 1024};
    std::vector<int> block_size = { 64};

    for (int BLOCK_DIM : block_size) {
        std::cout << "\n\n\n" << BLOCK_DIM<< "\n";
        for (int n : sizes) {
            double cpu_all = 0, gpu_all = 0, gpu_core = 0;
            float max_err = 0.0f;

            for (int run = 0; run < NUM_RUNS; ++run) {
                std::vector<float> h_vec(n, 0.0f);
                for (size_t i = 0; i < n; ++i)
                    h_vec[i] = dist(rng);

                double cpu_result;
                auto cpu_st = std::chrono::high_resolution_clock::now();
                cpu_sum(h_vec.data(), n, &cpu_result);
                cpu_all += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cpu_st).count();

                float* d_vec, * d_partial;
                size_t vec_sz = n * sizeof(float);
                int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
                size_t partial_sz = blocks * sizeof(float);

                cudaMalloc(&d_vec, vec_sz);
                cudaMalloc(&d_partial, partial_sz);

                // перенос с проца на девайс и начало первого замера для GPU
                auto gpu_st1 = std::chrono::high_resolution_clock::now();
                cudaMemcpy(d_vec, h_vec.data(), vec_sz, cudaMemcpyHostToDevice);

                // второй замер времени для GPU
                auto gpu_st2 = std::chrono::high_resolution_clock::now();
                gpu_sum <<< blocks, BLOCK_DIM, BLOCK_DIM * sizeof(float) >>> (d_vec, d_partial, n);
                cudaDeviceSynchronize();
                gpu_core += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - gpu_st2).count();

                // перенос с девайса на хост
                std::vector<float> h_partial(blocks);
                cudaMemcpy(h_partial.data(), d_partial, partial_sz, cudaMemcpyDeviceToHost);
                double gpu_result = 0.0;
                for (int b = 0; b < blocks; ++b)
                    gpu_result += h_partial[b];

                cudaFree(d_vec);
                cudaFree(d_partial);

                gpu_all += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - gpu_st1).count();

                // Ищем максимальную ошибку
                max_err = fmaxf(max_err, fabsf(static_cast<float>(cpu_result - gpu_result)));
            }
            // Берем среднее из полученного потому что есть NUM_RUNS
            cpu_all /= NUM_RUNS;
            gpu_all /= NUM_RUNS;
            gpu_core /= NUM_RUNS;

            std::cout << n << " | " << cpu_all << " | " << gpu_all << " | " << gpu_core << " | " << cpu_all / gpu_all << " | " << cpu_all / gpu_core << " | " << max_err << "\n";

        }
        
    }

    return 0;
}