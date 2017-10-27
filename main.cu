#include <ctime>
#include <vector>
#include <numeric>
#include <iostream>
#include <cuda_runtime.h>
#include "naive_kernel.hu"
#include "misc.hu"

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

    std::cout << 1/piest << std::endl;
    return 0;
}

double compute_naive(dim3 grid, dim3 block, unsigned int device,
		     unsigned int iterationsperThread)
{
    handleCudaErrors(cudaSetDevice(device));

    curandState *d_rngStates = 0;
    handleCudaErrors(cudaMalloc((void **) &d_rngStates, grid.x * block.x * sizeof(curandState)));

    unsigned int *d_res = 0;
    handleCudaErrors(cudaMalloc((void **) &d_res, grid.x * sizeof(unsigned int)));

    initRNG<<<grid, block>>>(d_rngStates, time(NULL));

    naive_kernel<<<grid, block,  block.x * sizeof(unsigned int)>>>(d_res, d_rngStates, iterationsperThread);

    std::vector<unsigned int> res(grid.x);
    handleCudaErrors(cudaMemcpy(&res[0], d_res, grid.x * sizeof(unsigned int),
				cudaMemcpyDeviceToHost));

    double estimate = static_cast<double>(std::accumulate(res.begin(),
							  res.end(), 0));
    estimate /= grid.x * block.x * iterationsperThread;

    cudaFree(d_rngStates);
    cudaFree(d_res);

    return estimate;
}
