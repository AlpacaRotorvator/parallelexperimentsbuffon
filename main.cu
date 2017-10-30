#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "naive_kernel.hu"
#include "batchrng_kernel.hu"
#include "misc.hu"

double compute_naive(dim3 grid, dim3 block, unsigned int device,
		     unsigned int iterationsperThread);

double compute_batchrng(dim3 grid, dim3 block, unsigned int device,
			unsigned int iterationsperThread,
			cudaDeviceProp *const deviceProp);

double compute_batchrng_2stream(dim3 grid, dim3 block, unsigned int device,
			unsigned int iterationsperThread,
			cudaDeviceProp *const deviceProp);


int main (int argc, char ** argv)
{
    unsigned int kernel = 0;
    double piest;
    cudaDeviceProp deviceProp;
    unsigned int device = 0;
    handleCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

    unsigned int iterationsPerThread = 1000 * 1000;
    dim3 grid = 16;
    dim3 block = 64;

    parseArgs(argc, argv, &iterationsPerThread, &deviceProp,
	      &grid.x, &block.x, &kernel);

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

    reportResults(piest, iterationsPerThread, grid.x, block.x, &deviceProp);

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
    handleCudaErrors(cudaSetDevice(device));
    //Set up the RNG
    using namespace std::placeholders;
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

    auto unifGen = std::bind(curandGenerateUniform, generator, _1, _2);

    //For partial results
    float *d_res = 0;
    handleCudaErrors(cudaMalloc((void **) &d_res, grid.x * sizeof(float)));

    //To calculate the final result
    double runningEstimate = 0;
    
    //Random number vector allocation strategy
    unsigned int numThreads = grid.x * block.x;
    unsigned long int totalSize = sizeof(float) * its * numThreads;
    unsigned int vecSize = 128 * 1024 * 1024;
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
	unifGen(d_angleVec, vecCount);
	unifGen(d_distVec, vecCount);

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

double compute_batchrng_2stream(dim3 grid, dim3 block, unsigned int device,
			unsigned int its,
			cudaDeviceProp *const deviceProp)
{
    handleCudaErrors(cudaSetDevice(device));
    //Set up the RNG
    using namespace std::placeholders;
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

    auto unifGen = std::bind(curandGenerateUniform, generator, _1, _2);

    //Set up our streams
    std::vector<cudaStream_t> streams (2);
    std::for_each(streams.begin(), streams.end(),
		  [] (cudaStream_t stream)
		  {
		      handleCudaErrors(cudaStreamCreate(&stream));
		  });
    //handleCudaErrors(cudaStreamCreate(&streams[0]));
    //handleCudaErrors(cudaStreamCreate(&streams[1]));
    

    //For partial results
    float *d_res = 0;
    handleCudaErrors(cudaMalloc((void **) &d_res, grid.x * sizeof(float)));

    float *d_res2 = 0;
    handleCudaErrors(cudaMalloc((void **) &d_res2, grid.x * sizeof(float)));

    //To calculate the final result
    double runningEstimate = 0;
    
    //Random number vector allocation strategy
    unsigned int numThreads = grid.x * block.x;
    unsigned long int totalSize = sizeof(float) * its * numThreads;
    unsigned int vecSize = 128 * 1024 * 1024;
    unsigned long int remainSize = totalSize;

    float * d_angleVec = 0;
    handleCudaErrors(cudaMalloc((void**) &d_angleVec, vecSize));

    float * d_distVec = 0;
    handleCudaErrors(cudaMalloc((void**) &d_distVec, vecSize));

    float * d_angleVec2 = 0;
    handleCudaErrors(cudaMalloc((void**) &d_angleVec2, vecSize));

    float * d_distVec2 = 0;
    handleCudaErrors(cudaMalloc((void**) &d_distVec2, vecSize));

    unsigned int vecCount = vecSize / sizeof(float);
    unsigned int numRuns = 0;
    std::vector<float> res(grid.x);

    //Here we go!
    while (remainSize > sizeof(float)) {
	numRuns++;
	if (remainSize < vecSize) {
	    vecCount = remainSize / sizeof(float);
	}

	curandSetStream(generator, streams[0]);
	unifGen(d_angleVec, vecCount);
	unifGen(d_distVec, vecCount);

	batchrng_kernel
	    <<<grid, block, block.x * sizeof(unsigned int), streams[0]>>>
	    ( d_res, d_angleVec, d_distVec, vecCount);

	handleCudaErrors(cudaMemcpyAsync(&res[0], d_res,
					 grid.x * sizeof(float),
					 cudaMemcpyDeviceToHost,
					 streams[0]));

	curandSetStream(generator, streams[1]);
	unifGen(d_angleVec2, vecCount);
	unifGen(d_distVec2, vecCount);

	batchrng_kernel<<<grid, block,
	    block.x * sizeof(unsigned int), streams[1]>>>
	    ( d_res2, d_angleVec2, d_distVec2, vecCount);

	runningEstimate += std::accumulate(res.begin(), res.end(), 0.0);

	cudaStreamSynchronize(streams[1]);
	handleCudaErrors(cudaMemcpy(&res[0], d_res2,
					 grid.x * sizeof(float),
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
    cudaFree(d_angleVec2);
    cudaFree(d_distVec2);
    cudaFree(d_res2);
    
    return runningEstimate / numRuns;
}
