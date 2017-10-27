#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "naive_kernel.hu"
#include "misc.h"

__device__ constexpr double pi2() { return acos(-1) / 2; };

__global__ void initRNG(curandState * const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

__device__ void draw(double &angle, double &distance, curandState &state)
{
    angle = cos(curand_uniform(&state) * pi2());
    distance = curand_uniform(&state) * 2;
}

__device__ double reduce_sum(unsigned int in)
{
    extern __shared__ unsigned int sdata[];

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] += sdata[ltid + s];
        }

        __syncthreads();
    }

    return sdata[0];
}

__global__ void naive_kernel (double *const results,
			      curandState *const rngStates,
			      const unsigned int numSims)
{
    // Determine thread ID
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int step = gridDim.x * blockDim.x;

    // Initialise the RNG
    curandState localState = rngStates[tid];

    // Count the number of draws that cross the line
    unsigned int pointsInside = 0;

    for (unsigned int i = 0; i < numSims ; i++)
    {
        precision angle;
        precision distance;
        draw(angle, distance, localState);

        if (distance <= angle)
        {
            pointsInside++;
        }
    }

    // Reduce within the block
    pointsInside = reduce_sum(pointsInside);

    // Store the result
    if (threadIdx.x == 0)
    {
        results[bid] = pointsInside / numSims;
    }
}
