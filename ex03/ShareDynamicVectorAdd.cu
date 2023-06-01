/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : nvcc vectorAdd.cu -o vectorAdd
 ============================================================================
 */
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <iostream>

static const int BLOCK_SIZE = 256;

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

//__global__ void dynamic_add(int *a, const int *b, const int n){
//    extern __shared__ int c[];
//    int id = blockIdx.x * blockDim.x + threadIdx.x;
//    int tid = threadIdx.x;
//    if (id < n){
//        c[tid] = a[id] + b[id];
//        __syncthreads();
//        a[id] = c[tid];
//    }
//}
__global__ void dynamic_add(int *a, const int *b, const int n){
    extern __shared__ int c[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (id < n){
        c[tid] = a[id] + b[id];
        __syncthreads();
        a[id] = c[tid];
    }

int main (void)
{
//  host (h) and device (d) pointers
    int *ha, *hb, *hc, *da, *db;
    int i, vector_size;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout<<"Enter the vector size: ";
    std::cin>>vector_size;

//  declare host vectors
    ha = new int[vector_size];
    hb = new int[vector_size];
    hc = new int[vector_size];

//  declare device vectors
    CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (int) * vector_size));
    CUDA_CHECK_RETURN (cudaMalloc ((void **) &db, sizeof (int) * vector_size));

//  initialize host vectors
    for (i = 0; i < vector_size; i++)
    {
        ha[i] = rand () % 10000;
        hb[i] = rand () % 10000;
    }

//  initialize device vectors
    CUDA_CHECK_RETURN (cudaMemcpy (da, ha, sizeof (int) * vector_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN (cudaMemcpy (db, hb, sizeof (int) * vector_size, cudaMemcpyHostToDevice));


//  define grid size
    int grid = ceil (vector_size * 1.0 / BLOCK_SIZE);
//  get initial time
    cudaEventRecord(start, nullptr);
//  allowing to use the 64KB of shared memory
//    cudaFuncSetAttribute(dynamic_add, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
//  call kernel
    dynamic_add <<< grid, BLOCK_SIZE, sizeof (int) * BLOCK_SIZE>>> (da, db, vector_size);
//  synchronize
    CUDA_CHECK_RETURN (cudaDeviceSynchronize ());
//  get end time
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
//  calculates the delta time
    cudaEventElapsedTime(&time, start, stop);

//  Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN (cudaGetLastError ());
//  copy result vector from device to host
    CUDA_CHECK_RETURN (cudaMemcpy (hc, da, sizeof (int) * vector_size, cudaMemcpyDeviceToHost));

//  validate the sum is correct
    for (i = 0; i < vector_size; i++)
    {
        if (hc[i] != ha[i] + hb[i]) {
            printf ("Error at index %i : %i VS %i\n", i, hc[i], ha[i] + hb[i]);
        }
    }
    //  write result into file
    printf("Delta time: %f\n", time);
    std::ofstream _file("../ex03/bitacora.txt", std::ios::app);
    _file << time << ",";
    _file.close();

//  free device vectors
    CUDA_CHECK_RETURN (cudaFree ((void *) da));
    CUDA_CHECK_RETURN (cudaFree ((void *) db));
//  free host vector
    delete[]ha;
    delete[]hb;
    delete[]hc;

//  destroy and clean resources
    CUDA_CHECK_RETURN (cudaDeviceReset ());

    return 0;
}
