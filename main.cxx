#include <cuda_runtime.h>
#include "misc.hxx"


int main (int argc, char * argv) {
    cudaDeviceProp deviceProp;
    handleCudaErrors(cudaGetDeviceProperties(&deviceProp));

    unsigned int iterationsPerThread = 1000 * 1000;
    dim3 grid = 16;
    dim3 block = 64;

    parseArgs(argc, &argv, &iterationsPerThread, deviceProp, grid.x, block.x);
    
    
    return 0;
}
