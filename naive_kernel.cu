#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include "cutils.hu"
#include "naive_kernel.hu"

__global__ void initRNG(curandState * const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

__device__ void draw(float &angle, float &distance, curandState &state)
{
    angle = cosf(curand_uniform(&state) * CUDART_PIO2_F);
    distance = curand_uniform(&state) * 2.0f;
}

__global__ void naive_kernel (float *const results,
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
        float angle;
        float distance;
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
        results[bid] = (static_cast<float>(numSims) * blockDim.x) /
	    (static_cast<float>(pointsInside) * gridDim.x);
    }
}
