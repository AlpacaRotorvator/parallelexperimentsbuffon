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

    for (unsigned int i = tid; i < numSims; i += step)
    {
        float angle = cosf(angleVec[i] * CUDART_PIO2_F);
        float distance = 2 * distVec[i];
	
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
        results[bid] = (static_cast<float>(numSims)) /
	    (static_cast<float>(pointsInside) * gridDim.x * gridDim.x);
    }
}
