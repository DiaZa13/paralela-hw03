/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : nvcc hello2.cu -o hello2 -arch=sm_20
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__global__ void hello() {

    int id = (blockIdx.y * gridDim.x + blockIdx.x) *
               blockDim.x * blockDim.y + threadIdx.y *
                                         blockDim.x + threadIdx.x;

    if (id == 9999)
        printf("\nDiana Zaray Corado #191025 %i\n", id);
}

int main() {
    dim3 g(25, 5);
    dim3 b(32, 25);
    hello <<< g, b >>>();

    cudaDeviceSynchronize();
    return 0;
}
