#include <iostream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_timer.h>
#include "naive_kernel.hu"
#include "batchrng_kernel.hu"
#include "misc.hu"

double compute_naive(dim3 grid, dim3 block, unsigned int device,
		     unsigned int iterationsperThread);

double compute_batchrng(dim3 grid, dim3 block, unsigned int device,
			unsigned int iterationsperThread,
			cudaDeviceProp *const deviceProp);


int main (int argc, char ** argv)
{
    unsigned int kernel = 0;
    double piest;
    cudaDeviceProp deviceProp;
    unsigned int device = 0;
    handleCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
    handleCudaErrors(cudaSetDevice(device));

    unsigned int iterationsPerThread = 1000 * 1000;
    dim3 grid = 16;
    dim3 block = 64;

    parseArgs(argc, argv, &iterationsPerThread, &deviceProp,
	      &grid.x, &block.x, &kernel);

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    switch (kernel) {
    case 0:
	piest = compute_naive(grid, block, device, iterationsPerThread);
	break;
    case 1:
	piest = compute_batchrng(grid, block, device, iterationsPerThread,
				 &deviceProp);
	break;
    case 2:
	piest = compute_batchrng(grid, block, device,
					 iterationsPerThread, &deviceProp);
    }
    sdkStopTimer(&timer);
    float elapsedTime = sdkGetAverageTimerValue(&timer)/1000.0f;
    
    reportResults(piest, iterationsPerThread, grid.x, block.x, &deviceProp, elapsedTime);

    return 0;
}

double compute_naive(dim3 grid, dim3 block, unsigned int device,
		     unsigned int iterationsperThread)
{
    handleCudaErrors(cudaSetDevice(device));

    curandState *d_rngStates = 0;
    handleCudaErrors(cudaMalloc((void **) &d_rngStates, grid.x * block.x * sizeof(curandState)));

    float *d_res = 0;
    handleCudaErrors(cudaMalloc((void **) &d_res, grid.x * sizeof(float)));

    initRNG<<<grid, block>>>(d_rngStates, time(NULL));

    naive_kernel<<<grid, block,  block.x * sizeof(unsigned int)>>>(d_res, d_rngStates, iterationsperThread);

    std::vector<float> res(grid.x);
    handleCudaErrors(cudaMemcpy(&res[0], d_res, grid.x * sizeof(float),
				cudaMemcpyDeviceToHost));

    double estimate = std::accumulate(res.begin(), res.end(), 0.0);
    cudaFree(d_rngStates);
    cudaFree(d_res);

    return estimate;
}

double compute_batchrng(dim3 grid, dim3 block, unsigned int device,
			unsigned int its,
			cudaDeviceProp *const deviceProp)
{
    //Set up the RNG
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

    //For partial results
    float *d_res = 0;
    handleCudaErrors(cudaMalloc((void **) &d_res, grid.x * sizeof(float)));

    //To calculate the final result
    double runningEstimate = 0;
    
    //Random number vector allocation strategy
    unsigned int numThreads = grid.x * block.x;
    //Total size of *1* vector, I need *2*
    unsigned long int totalSize = sizeof(float) * its * numThreads;

    //Get device's free and total global memory
    size_t freeMem = 0;
    size_t totalMem = 0;
    handleCudaErrors(cudaMemGetInfo(&freeMem, &totalMem));
    unsigned int vecSize = 0;
    
    //Allocate everything at once if we can get away with it
    if (2 * totalSize <= freeMem * 0.9) {
	vecSize = totalSize;
    }
    else {
	//Spare 10% of the device's free memory(not because this program will need it, but because I have only one GPU and I don't feel like locking my system)
	vecSize = static_cast<unsigned int>(freeMem * 0.9 / 2);
    }
    
    unsigned long int remainSize = totalSize;

    float * d_angleVec = 0;
    handleCudaErrors(cudaMalloc((void**) &d_angleVec, vecSize));

    float * d_distVec = 0;
    handleCudaErrors(cudaMalloc((void**) &d_distVec, vecSize));


    unsigned int vecCount = vecSize / sizeof(float);
    unsigned int numRuns = 0;
    std::vector<float> res(grid.x);

    unsigned int count = 0;
    //Here we go!
    while (remainSize > sizeof(float)) {
	numRuns++;
	if (remainSize < vecSize) {
	    vecCount = remainSize / sizeof(float);
	}
	count += vecCount;
	curandGenerateUniform(generator, d_angleVec, vecCount);
	curandGenerateUniform(generator, d_distVec, vecCount);

	batchrng_kernel<<<grid, block,  block.x * sizeof(unsigned int)>>>
	    ( d_res, d_angleVec, d_distVec, vecCount);

	handleCudaErrors(cudaMemcpy(&res[0], d_res, grid.x * sizeof(float),
				    cudaMemcpyDeviceToHost));
	runningEstimate += std::accumulate(res.begin(), res.end(), 0.0);
	
	if (remainSize > vecSize) {
	    remainSize -= vecSize;
	}
	else {
	    break;
	}
    }
    
    cudaFree(d_angleVec);
    cudaFree(d_distVec);
    cudaFree(d_res);
    
    return runningEstimate / numRuns;
}
