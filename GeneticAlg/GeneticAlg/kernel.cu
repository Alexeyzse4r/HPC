#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/sequence.h>

#define POLY_SIZE 5

// есть возможность выбора типа ошибки потому что в доке написано максимальная но она работает существенно хуже
//#define USE_MAX_ERROR
#define USE_MSE 


constexpr int MAX_GEN = 2000;
constexpr int MAX_CONST_GEN = 50;
constexpr double AIMED_FITNESS = 1e-6;

struct Individual {
    float coeffs[POLY_SIZE];
};

struct Function_data {
    std::vector<double> x;
    std::vector<double> y;
};


__global__ void setup_rng(curandState* states, int seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        curand_init(seed, i, 0, &states[i]);
}

__global__ void init_population(
    Individual* pop,
    curandState* states,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    curandState st = states[i];

    for (int k = 0; k < POLY_SIZE; k++) {
        float r = curand_uniform(&st);
        pop[i].coeffs[k] = r * 2.f - 1.f;
    }

    states[i] = st;
}

__global__ void fitness_kernel(
    Individual* pop,
    const double* x,
    const double* y,
    double* fit,
    int n_points,
    int pop_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop_size) return;

#ifdef USE_MAX_ERROR
    double max_err = 0.0;

    for (int p = 0; p < n_points; p++) {
        double xi = x[p];
        double yi = y[p];

        double pred = 0.0;
        double xp = 1.0;

        for (int k = 0; k < POLY_SIZE; k++) {
            pred += pop[i].coeffs[k] * xp;
            xp *= xi;
        }

        double err_val = fabs(pred - yi);
        if (err_val > max_err) max_err = err_val;
    }

    fit[i] = max_err;
#endif
#ifdef USE_MSE
    double sum_sq = 0.0;

    for (int p = 0; p < n_points; p++) {
        double xi = x[p];
        double yi = y[p];

        double pred = 0.0;
        double xp = 1.0;

        for (int k = 0; k < POLY_SIZE; k++) {
            pred += pop[i].coeffs[k] * xp;
            xp *= xi;
        }

        double err = pred - yi;
        sum_sq += err * err;
    }

    fit[i] = sum_sq;
#endif
}
__global__ void crossover(
    Individual* pop,
    Individual* child,
    const int* idx,
    curandState* states,
    int n,
    int elite
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i < elite) {
        child[i] = pop[idx[i]];
        return;
    }

    curandState st = states[i];

    int p1 = idx[curand(&st) % (n / 3)];
    int p2 = idx[curand(&st) % (n / 3)];

    int cut = 1 + curand(&st) % (POLY_SIZE - 1);

    for (int k = 0; k < POLY_SIZE; k++) {
        child[i].coeffs[k] =
            (k < cut) ? pop[p1].coeffs[k] : pop[p2].coeffs[k];
    }

    states[i] = st;
}
__global__ void mutation(
    Individual* pop,
    curandState* states,
    int n,
    float Em,
    float Dm
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    curandState st = states[i];

    float sigma = sqrtf(Dm);

    for (int k = 0; k < POLY_SIZE; k++) {
        float r = curand_uniform(&st);
        if (r < 0.05f) {
            float noise = Em + sigma * curand_normal(&st);
            pop[i].coeffs[k] += noise * 0.1f;
        }
    }

    states[i] = st;
}
bool GA_GPU(
    const Function_data& data,
    int N,
    int P,
    float Em,
    float Dm, 
    std::vector<double>& bestC,
    double& bestFit,
    double& timeSec,
    int& generations
) {
    thrust::device_vector<Individual> pop(P), child(P);
    thrust::device_vector<double> fit(P);
    thrust::device_vector<int> idx(P);

    double* dx, * dy;
    curandState* states;

    cudaMalloc(&dx, N * sizeof(double));
    cudaMalloc(&dy, N * sizeof(double));
    cudaMalloc(&states, P * sizeof(curandState));

    cudaMemcpy(dx, data.x.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, data.y.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    int T = 256;
    int B = (P + T - 1) / T;

    setup_rng << <B, T >> > (states, time(nullptr), P);
    init_population << <B, T >> > (
        thrust::raw_pointer_cast(pop.data()),
        states,
        P
        );

    cudaDeviceSynchronize();

    double prevBest = 1e30;
    int noImprove = 0;
    generations = 0;

    auto start = std::chrono::steady_clock::now();

    for (int g = 0; g < MAX_GEN; g++) {

        fitness_kernel << <B, T >> > (
            thrust::raw_pointer_cast(pop.data()),
            dx, dy,
            thrust::raw_pointer_cast(fit.data()),
            N, P
            );

        cudaDeviceSynchronize();

        thrust::sequence(idx.begin(), idx.end());
        thrust::sort_by_key(fit.begin(), fit.end(), idx.begin());

        cudaMemcpy(&bestFit,
            thrust::raw_pointer_cast(fit.data()),
            sizeof(double),
            cudaMemcpyDeviceToHost);

        if (bestFit < AIMED_FITNESS)
            break;

        if (fabs(bestFit - prevBest) < 1e-12)
            noImprove++;
        else
            noImprove = 0;

        if (noImprove >= MAX_CONST_GEN)
            break;

        prevBest = bestFit;

        int elite = P * 0.05;

        crossover << <B, T >> > (
            thrust::raw_pointer_cast(pop.data()),
            thrust::raw_pointer_cast(child.data()),
            thrust::raw_pointer_cast(idx.data()),
            states,
            P,
            elite
            );

        std::swap(pop, child);

        mutation << <B, T >> > (
            thrust::raw_pointer_cast(pop.data()),
            states,
            P,
            Em,
            Dm
            );

        cudaDeviceSynchronize();

        generations++;
    }

    auto end = std::chrono::steady_clock::now();
    timeSec = std::chrono::duration<double>(end - start).count();

    std::vector<int> h_idx(P);
    std::vector<Individual> h_pop(P);

    cudaMemcpy(h_idx.data(),
        thrust::raw_pointer_cast(idx.data()),
        P * sizeof(int),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(h_pop.data(),
        thrust::raw_pointer_cast(pop.data()),
        P * sizeof(Individual),
        cudaMemcpyDeviceToHost);

    Individual best = h_pop[h_idx[0]];

    bestC.resize(POLY_SIZE);
    for (int i = 0; i < POLY_SIZE; i++)
        bestC[i] = best.coeffs[i];

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(states);

    return true;
}

double f_true(double x) {
    return 1.0 + 2.0 * x + 0.5 * x * x + 0.1 * x * x * x + 1.0 * x * x * x * x;
}

double fitness_cpu(const Individual& ind, const Function_data& data) {
#ifdef USE_MAX_ERROR
    double max_err = 0.0;

    for (int i = 0; i < data.x.size(); i++) {
        double x = data.x[i];
        double y = data.y[i];

        double pred = 0.0;
        double xp = 1.0;

        for (int k = 0; k < POLY_SIZE; k++) {
            pred += ind.coeffs[k] * xp;
            xp *= x;
        }

        double err_val = fabs(pred - y);
        if (err_val > max_err) max_err = err_val;
    }

    return max_err;
#endif

#ifdef USE_MSE
    double sum_sq = 0.0;

    for (int i = 0; i < data.x.size(); i++) {
        double x = data.x[i];
        double y = data.y[i];

        double pred = 0.0;
        double xp = 1.0;

        for (int k = 0; k < POLY_SIZE; k++) {
            pred += ind.coeffs[k] * xp;
            xp *= x;
        }

        double err = pred - y;
        sum_sq += err * err;
    }

    return sum_sq;
#endif
}



bool GA_CPU(
    const Function_data& data,
    int P,
    float Em,
    float Dm,
    std::vector<double>& bestC,
    double& bestFit,
    double& timeSec,
    int& generations
) {
    std::mt19937 gen(time(nullptr));
    std::uniform_real_distribution<double> r(-1.0, 1.0);
    std::normal_distribution<double> noise(Em, sqrt(Dm));

    std::vector<Individual> pop(P), newPop(P);
    std::vector<double> fit(P);
    for (auto& ind : pop) {
        for (int i = 0; i < POLY_SIZE; i++) {
            ind.coeffs[i] = r(gen);
        }
    }

    double prevBest = 1e30;
    int noImprove = 0;
    generations = 0;

    auto start = std::chrono::steady_clock::now();

    for (int g = 0; g < MAX_GEN; g++) {


        for (int i = 0; i < P; i++) {
            fit[i] = fitness_cpu(pop[i], data);
        }
        std::vector<int> idx(P);
        for (int i = 0; i < P; i++) {
            idx[i] = i;
        }

        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return fit[a] < fit[b]; });

        bestFit = fit[idx[0]];

        if (bestFit < AIMED_FITNESS)
            break;

        if (fabs(bestFit - prevBest) < 1e-12)
            noImprove++;
        else
            noImprove = 0;

        if (noImprove >= MAX_CONST_GEN)
            break;
        prevBest = bestFit;

        int elite = P * 0.05;
        for (int i = 0; i < elite; i++) {
            newPop[i] = pop[idx[i]];
        }

        for (int i = elite; i < P; i++) {
            int p1 = idx[gen() % (P / 3)];
            int p2 = idx[gen() % (P / 3)];

            int cut = 1 + (gen() % (POLY_SIZE - 1));

            for (int k = 0; k < POLY_SIZE; k++) {
                newPop[i].coeffs[k] =
                    (k < cut) ? pop[p1].coeffs[k] : pop[p2].coeffs[k];
            }

            for (int k = 0; k < POLY_SIZE; k++) {
                if (gen() / (double)std::mt19937::max() < 0.05f) {
                    newPop[i].coeffs[k] += noise(gen) * 0.1f;
                }
            }
        }

        pop = newPop;
        generations++;
    }

    auto end = std::chrono::steady_clock::now();
    timeSec = std::chrono::duration<double>(end - start).count();

    bestC.resize(POLY_SIZE);
    for (int i = 0; i < POLY_SIZE; i++)
        bestC[i] = pop[0].coeffs[i];

    return true;
}


int main() {

    int N = 1000;
    int P = 2000;

#ifdef USE_MAX_ERROR
    std::cout << "MAX_ERROR\n";
#endif
#ifdef USE_MSE
    std::cout << "MSE\n";
#endif

    float Em = 0.0f;
    float Dm = 1.0f;

    Function_data data;
    data.x.resize(N);
    data.y.resize(N);

    for (int i = 0; i < N; i++) {
        double x = -1.0 + 2.0 * i / (double)(N - 1);
        data.x[i] = x;
        data.y[i] = f_true(x);
    }

    std::vector<double> cpuC, gpuC;
    double cpuFit, gpuFit;
    double cpuTime, gpuTime;
    int cpuGenerations, gpuGenerations;

    GA_CPU(data, P, Em, Dm, cpuC, cpuFit, cpuTime, cpuGenerations);
    GA_GPU(data, N, P, Em, Dm, gpuC, gpuFit, gpuTime, gpuGenerations);

    std::cout << "\n\n\nCPU\n";
    std::cout << "Generations: " << cpuGenerations << "\n";
    std::cout << "Time: " << cpuTime << "\n";
    std::cout << "Fitness: " << cpuFit << "\n";

    std::cout << "\nCPU Coeffs:\n";
    for (int i = 0; i < POLY_SIZE; i++)
        std::cout << "c" << i << " = " << cpuC[i] << "\n";

    std::cout << "\n\n\nGPU\n";
    std::cout << "Generations: " << gpuGenerations << "\n";
    std::cout << "Time: " << gpuTime << "\n";
    std::cout << "Fitness: " << gpuFit << "\n";

    std::cout << "\nGPU Coeffs:\n";
    for (int i = 0; i < POLY_SIZE; i++)
        std::cout << "c" << i << " = " << gpuC[i] << "\n";
}