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

__device__ float reduce_sum(float in)
{
    extern __shared__ float sdataF[];

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdataF[ltid] = in;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdataF[ltid] += sdataF[ltid + s];
        }

        __syncthreads();
    }

    return sdataF[0];
}
