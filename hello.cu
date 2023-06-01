/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : nvcc hello.cu -o hello -arch=sm_20
 ============================================================================
 */
#include <cstdio>
#include <cuda.h>

__global__ void hello()
{
    int thread = (blockIdx.x) * blockDim.x + threadIdx.x;
    printf("Thread -> %d, Diana Zaray Corado #191025\n",thread);
}

int main()
{
    hello<<<1,2048>>>();
    cudaThreadSynchronize(); //deprecated
    return 0;
}
