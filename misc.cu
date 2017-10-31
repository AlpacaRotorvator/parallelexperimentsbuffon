#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

using namespace std;

void handleCudaErrors (cudaError_t cudaResult, string msg)
{
    if (cudaResult != cudaSuccess) {
	msg += cudaGetErrorString(cudaResult);
	throw runtime_error(msg);
    }
}

void printHelpmsg ()
{
    string helpMsg = "Usage: buffoncuda [-n <NUMINT>] [-b <BLOCKNUM>] [-t <TNUM>] [-k <KERNID>]\n\n";
    helpMsg += "Please remember me to finish writing this if you feel frustrated by the lack of proper documentation.\n";
    cout << helpMsg;
    exit(0);
}

void parseArgs (int argc, char ** argv, unsigned int *  iterationsPerThread,
		cudaDeviceProp * const deviceProp, unsigned int * numBlocks,
		unsigned int *  threadsPerBlock, unsigned int * kernel)
{
    char cmdFlag;
    int candidate = 0;

    while((cmdFlag = getopt(argc, argv, "n:b:t:k:h")) != -1) {
	switch (cmdFlag)
	    {
	    case 'n':
		*iterationsPerThread = atoi(optarg);
		break;
	    case 'b':
		candidate = atoi(optarg);
		if (candidate <= 0) {
		    throw runtime_error("Number of blocks must be greater or equal than zero");
		}
		else {
		    *numBlocks = candidate;
		}
		break;
	    case 't':
		candidate = atoi(optarg);
		if (candidate <= 0) {
		    throw runtime_error("Number of threads per block must be greater or equal than zero.");
		}
		else if ((candidate & (candidate - 1)) != 0) {
		    throw runtime_error("Number of threads per block must be a power of two(for efficient reduction).");
		}
		else {
		    *threadsPerBlock = candidate;
		}
		break;
	    case 'k':
		candidate = atoi(optarg);
		if (candidate < 0 || candidate > 2) {
		    throw runtime_error("Kernel number must be 0, 1 or 2");
		}
		else {
		    *kernel = candidate;
		}
		break;
	    case 'h':
		printHelpmsg();
		break;
	    }
    }
}

void reportResults (double estimate, unsigned int itpT,  unsigned int gridS,
		    unsigned int blockS, cudaDeviceProp *const deviceProp, float elapsedTime)
{
    double abserr = abs(estimate - CUDART_PI);
    double relerr = abserr / CUDART_PI;

    cout << "      RESULTS:          " << endl;
    cout << "========================" << endl;
    cout << "Device Name:            " << deviceProp->name << endl;
    cout << "Grid Size:              " << gridS << endl;
    cout << "Block Size:             " << blockS << endl;
    cout << "Number of threads:      " << blockS * gridS << endl;
    cout << "Iterations per thread:  " << itpT << endl;
    cout << "Total iterations:       " << static_cast<double>(itpT) * blockS * gridS << endl;
    cout << "Kernel execution time:  " << elapsedTime << "s" << endl;
    cout << "PI estimate:            " << estimate << endl;
    cout << "Abolute error:          " << abserr << endl;
    cout << "Relative error:         " << relerr << endl;
    
}
