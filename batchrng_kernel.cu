#include <cuda_runtime.h>
#include <math_constants.h>
#include "batchrng_kernel.hu"

__global__ void batchrng_kernel (float *const results,
				 float *const angleVec,
				 float *const distVec,
				 const unsigned int numSims)
{
    // Determine thread ID
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Count the number of draws that cross the line
    unsigned int pointsInside = 0;

    for (unsigned int i = 0; i < numSims / step; i++)
    {
        float angle = cosf(angleVec[tid + i * step] * CUDART_PIO2_F);
        float distance = 2 * distVec[tid + i * step];
	
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
