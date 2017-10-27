#include <ctime>
#include <cuda_runtime.h>
#include "naive_kernel.hu"
#include "misc.hu"

constexpr double pi() { return acos(-1); };

double compute_naive(dim3 grid, dim3 block, unsigned int device,
		     unsigned int iterationsperThread);

int main (int argc, char ** argv)
{
    double piest;
    cudaDeviceProp deviceProp;
    unsigned int device = 0;
    handleCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

    unsigned int iterationsPerThread = 1000 * 1000;
    dim3 grid = 16;
    dim3 block = 64;

    parseArgs(argc, argv, &iterationsPerThread, &deviceProp, &grid.x, &block.x);

    piest = compute_naive(grid, block, device, iterationsPerThread);

    cout << piest;
    return 0;
}

double compute_naive(dim3 grid, dim3 block, unsigned int device,
		     unsigned int iterationsperThread)
{
    handleCudaErrors(cudaSetDevice(device));

    curandState *d_rngStates = 0;
    handleCudaErrors(cudaMalloc((void **) &d_rngStates, grid.x * block.x * sizeof(curandState)));

    double *d_res = 0;
    handleCudaErrors(cudaMalloc((void **) &d_res, grid.x * sizeof(double)));

    initRNG<<<grid, block>>>(d_rngStates, time(NULL));

    naive_kernel<<<grid, block>>>(d_res, d_rngStates, iterationsperThread);

    std::vector<double> res(grid.x);
    handleCudaErrors(cudaMemcpy(&res[0], d_res, grid.x * sizeof(double),
				cudaMemcpyDeviceToHost));

    double estimate = std::accumulate(res.begin(), res.end(), 0);
    estimate /= grid.x * block.x;

    cudaFree(d_rngStates);
    cudaFree(d_res);

    return estimate;
}
