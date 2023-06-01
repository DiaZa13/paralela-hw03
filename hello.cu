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
//   int myID = ( blockIdx.z * gridDim.x * gridDim.y  +
//                blockIdx.y * gridDim.x +
//                blockIdx.x ) * blockDim.x * blockDim.y * blockDim.z +
//                threadIdx.z *  blockDim.x * blockDim.y +
//                threadIdx.y * blockDim.x +
//                threadIdx.x;

//  Simplificando...
    //grid: 3D --- z,y,x: all dims and blockids
    //block: 1D -- x
    int myID = (blockIdx.y * gridDim.x + blockIdx.x) *
               blockDim.x * blockDim.y + threadIdx.y *
                                         blockDim.x + threadIdx.x;

    printf("Diana Zaray Corado #191025 %i\n", myID);
}

int main() {
    dim3 g(4, 2);
    dim3 b(32, 16);
    hello <<< g, b >>>();
//    cudaThreadSynchronize ();
    cudaDeviceSynchronize();  //use instead, ^ is deprecated
    return 0;
}
