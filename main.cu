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
#include "cudaResourceWrapper.hu"

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

    CudaResWrapper<curandState> d_rngStates(grid.x * block.x);

    CudaResWrapper<float> d_res(grid.x);

    initRNG<<<grid, block>>>(d_rngStates.getPtr(), time(NULL));

    naive_kernel<<<grid, block,  block.x * sizeof(unsigned int)>>>
	(d_res.getPtr(), d_rngStates.getPtr(), iterationsperThread);

    std::vector<float> res(grid.x);
    handleCudaErrors(cudaMemcpy(&res[0], d_res.getPtr(), grid.x * sizeof(float),
				cudaMemcpyDeviceToHost));

    double estimate = std::accumulate(res.begin(), res.end(), 0.0);

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
    CudaResWrapper<float> d_res(grid.x);


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
    size_t vecSize = 0;
    
    //Allocate everything at once if we can get away with it
    if (2 * totalSize <= freeMem * 0.9) {
	vecSize = totalSize;
    }
    else {
	//Spare 10% of the device's free memory(not because this program will need it, but because I have only one GPU and I don't feel like locking my system)
	vecSize = static_cast<unsigned int>(freeMem * 0.9 / 2 );
    }
    size_t vecCount = vecSize / sizeof(float);
    size_t remainSize = totalSize;

    CudaResWrapper<float> d_angleVec(vecCount);
    CudaResWrapper<float> d_distVec(vecCount);

    unsigned int numRuns = 0;
    std::vector<float> res(grid.x);

    //Here we go!
    while (remainSize > sizeof(float)) {
	numRuns++;
	if (remainSize < vecSize) {
	    vecCount = remainSize / sizeof(float);
	}
	
	curandGenerateUniform(generator, d_angleVec.getPtr(), vecCount);
	curandGenerateUniform(generator, d_distVec.getPtr(), vecCount);

	batchrng_kernel<<<grid, block,  block.x * sizeof(unsigned int)>>>
	    ( d_res.getPtr(), d_angleVec.getPtr(), d_distVec.getPtr(), vecCount);

	handleCudaErrors(cudaMemcpy(&res[0], d_res.getPtr(),
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
        
    return runningEstimate / numRuns;
}
