#include <iostream>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

void handleCudaErrors (cudaError_t cudaResult, std::string msg = "Error: ") {
    if (cudaResult != cudaSuccess) {
	msg += cudaGetErrorString(cudaResult);
	throw std::runtime_error(msg);
    }
}

void printHelpmsg () {
    std::string helpMsg = "Usage: buffoncuda [-n <NUMINT>] [-b <BLOCKNUM>] [-t <TNUM>]";
    std::cout << helpMsg;
}

void parseArgs (int argc, char ** argv, unsigned int *  iterationsPerThread,
		cudaDeviceProp * const deviceProp, unsigned int * numBlocks,
		unsigned int *  threadsPerBlock) {
    char cmdFlag;

    while((cmdFlag = getopt(argc, argv, "n:b:t:h")) != -1) {
	switch (cmdFlag)
	    {
	    case 'n':
		*iterationsPerThread = atoi(optarg);
		break;
	    case 'b':
		int candidate = atoi(optarg);
		if (candidate <= 0) {
		    throw std::runtime_error("Number of blocks must be greater or equal than zero");
		}
		else {
		    *numBlocks = candidate;
		}
		break;
	    case 't':
		int candidate = atoi(optarg);
		if (candidtate <= 0) {
		    throw std::runtime_error("Number of threads per block must be greater or equal than zero.");
		}
		else if ((n & (n - 1)) != 0) {
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
