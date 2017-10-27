#include <string>
#include <cuda_runtime.h>

void handleCudaErrors (cudaError_t cudaResult, std::string msg);

void printHelpmsg ();

void parseArgs (int argc, char ** argv, unsigned int *  iterationsPerThread,
		cudaDeviceProp * const deviceProp, unsigned int * numBlocks,
		unsigned int *  threadsPerBlock);
