# parallelexperimentsbuffon

Buffon's Needle. In Cuda.

# Building
With cmake:

    mkdir build && cd build
    cmake .. && make

Without cmake:

    make

# Running

    ./buffoncuda [STUFF]

Without additional arguments, runs a million throws per thread on a 16 block long grid with 64 threads per block(completely arbitrary choice) 
using the naive kernel(see below for details on kernels)
   
## Command line parameters

* `-n x`: run x throws of the needle per thread.
* `-b x`: use a grid with x blocks.
* `-t x`: use x threads per block.
* `-k x`: currently 0 for naive kernel, 1 for batchRNG kernel. See below for details.

## Available kernels
0. naive: almost what you get if you pick a serial implementation and throw \_\_global\_\_ around generously.
1. batchRNG: allocates blocks(currently hardcoded 2x128MB) of global memory, fill them up with random numbers and 
let the threads loose on them. Actually slower than naive because I have no idea how to deal with global memory efficiently.
