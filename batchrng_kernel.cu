#include <cuda_runtime.h>
#include <math_constants.h>

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
        float distance = 2 * distleVec[tid + i * step];
	
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

__device__ unsigned int reduce_sum(unsigned int in)
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
