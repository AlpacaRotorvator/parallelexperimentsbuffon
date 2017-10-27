#include <iostream>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include <cuda_runtime.h>

void handleCudaErrors (cudaError_t cudaResult, std::string msg) {
    if (cudaResult != cudaSuccess) {
	msg += cudaGetErrorString(cudaResult);
	throw std::runtime_error(msg);
    }
}

void printHelpmsg () {
    std::string helpMsg = "Usage: buffoncuda [-n <NUMINT>] [-b <BLOCKNUM>] [-t <TNUM>]\n";
    std::cout << helpMsg;
}

void parseArgs (int argc, char ** argv, unsigned int *  iterationsPerThread,
		cudaDeviceProp * const deviceProp, unsigned int * numBlocks,
		unsigned int *  threadsPerBlock) {
    char cmdFlag;
    int candidate = 0;

    while((cmdFlag = getopt(argc, argv, "n:b:t:h")) != -1) {
	switch (cmdFlag)
	    {
	    case 'n':
		*iterationsPerThread = atoi(optarg);
		break;
	    case 'b':
		candidate = atoi(optarg);
		if (candidate <= 0) {
		    throw std::runtime_error("Number of blocks must be greater or equal than zero");
		}
		else {
		    *numBlocks = candidate;
		}
		break;
	    case 't':
		candidate = atoi(optarg);
		if (candidate <= 0) {
		    throw std::runtime_error("Number of threads per block must be greater or equal than zero.");
		}
		else if ((candidate & (candidate - 1)) != 0) {
		    throw std::runtime_error("Number of threads per block must be a power of two(for efficient reduction).");
		}
		else {
		    *threadsPerBlock = candidate;
		}
		break;
	    case 'h':
		printHelpmsg();
		break;
	    }
    }
}
