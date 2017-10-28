#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

using namespace std;

void handleCudaErrors (cudaError_t cudaResult, string msg) {
    if (cudaResult != cudaSuccess) {
	msg += cudaGetErrorString(cudaResult);
	throw runtime_error(msg);
    }
}

void printHelpmsg () {
    string helpMsg = "Usage: buffoncuda [-n <NUMINT>] [-b <BLOCKNUM>] [-t <TNUM>]\n\n";
    helpMsg += "Please remember me to finish writing this if you feel frustrated by the lack of proper documentation.\n";
    cout << helpMsg;
    exit(0);
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
	    case 'h':
		printHelpmsg();
		break;
	    }
    }
}

void reportResults (double estimate, unsigned int itpT,  unsigned int gridS,
		    unsigned int blockS, cudaDeviceProp *const deviceProp)
{
    double abserr = abs(estimate - CUDART_PI);
    double relerr = abserr / CUDART_PI;

    cout << "      RESULTADOS:       " << endl;
    cout << "========================" << endl;
    cout << "Nome do dispositivo:    " << deviceProp->name << endl;
    cout << "Tamanho da grid:        " << gridS << endl;
    cout << "Tamanho dos blocos:     " << blockS << endl;
    cout << "Número de threads:      " << blockS * gridS << endl;
    cout << "Estimativas por thread: " << itpT << endl;
    cout << "Total de estimativas:   " << static_cast<double>(itpT) * blockS * gridS << endl;
    cout << "Estimativa de PI:       " << estimate << endl;
    cout << "Erro absoluto:          " << abserr << endl;
    cout << "Erro relativo:          " << relerr << endl;
    
}
