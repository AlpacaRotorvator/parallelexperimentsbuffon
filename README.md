# parallelexperimentsbuffon

Buffon's needle as a platform for exploring performance (un)optimizations in CUDA.

# Building

    make

# Running

    ./buffoncuda [STUFF]

Without additional arguments, runs a million throws per thread on a 16 block long grid with 64 threads per block(completely arbitrary, unoptimal, choice) 
using the naive kernel(see below for details on kernels)
   
## Command line parameters

* `-n x`: run x throws of the needle per thread.
* `-b x`: use a grid with x blocks.
* `-t x`: use x threads per block.
* `-k x`: currently 0 for naive kernel, 1 for batchRNG kernel. See below for details.

## Available kernels
0. naivest: almost what you get if you pick a serial implementation and throw \_\_global\_\_ around generously.
1. naive: same as 0, but using cospif for ~70% increased performance and some accuracy gains.
2. batchRNG: allocates large blocks of global memory, fills them up with random numbers and lets the threads loose on them. Actually slower than naive because I have no idea how to deal with global memory efficiently.
