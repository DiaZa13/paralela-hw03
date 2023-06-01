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
#include <iostream>
#include <fstream>


static const int BLOCK_SIZE = 256;
static const int N = 12000;

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

__global__ void vadd (const int *a, const int *b, int *c, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
//  every thread just work on his id
    if (id < N)
        c[id] = a[id] + b[id];
}

int main (void)
{
//  host (h) and device (d) pointers
    int *ha, *hb, *hc, *da, *db, *dc;
    int i;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//  declare host vectors
    ha = new int[N];
    hb = new int[N];
    hc = new int[N];

//  declare device vectors
    CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (int) * N)); //bloquea
    CUDA_CHECK_RETURN (cudaMalloc ((void **) &db, sizeof (int) * N));
    CUDA_CHECK_RETURN (cudaMalloc ((void **) &dc, sizeof (int) * N));

//  initialize host vectors
    for (i = 0; i < N; i++)
    {
        ha[i] = rand () % 10000;
        hb[i] = rand () % 10000;
    }

//  initialize device vectors
    CUDA_CHECK_RETURN (cudaMemcpy (da, ha, sizeof (int) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN (cudaMemcpy (db, hb, sizeof (int) * N, cudaMemcpyHostToDevice));

//  define grid size
    int grid = ceil (N * 1.0 / BLOCK_SIZE);
//  get initial time
    cudaEventRecord(start, nullptr);
//  call kernel
    vadd <<< grid, BLOCK_SIZE >>> (da, db, dc, N);
//  synchronize
    CUDA_CHECK_RETURN (cudaDeviceSynchronize ());
//  get end time
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
//  calculates the delta time
    cudaEventElapsedTime(&time, start, stop);

    // Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN (cudaGetLastError ());
//  copy result vector from device to host
    CUDA_CHECK_RETURN (cudaMemcpy (hc, dc, sizeof (int) * N, cudaMemcpyDeviceToHost));

//  validate the sum is correct
    for (i = 0; i < N; i++)
    {
        if (hc[i] != ha[i] + hb[i])
            printf ("Error at index %i : %i VS %i\n", i, hc[i], ha[i] + hb[i]);
    }
    printf("Delta time: %f\n", time);
//  write result into file
    std::ofstream _file("../ex03/bitacora.txt", std::ios::app);
    _file << time << ",";
    _file.close();

//  free device vectors
    CUDA_CHECK_RETURN (cudaFree ((void *) da));
    CUDA_CHECK_RETURN (cudaFree ((void *) db));
    CUDA_CHECK_RETURN (cudaFree ((void *) dc));
//  free host vector
    delete[]ha;
    delete[]hb;
    delete[]hc;

    //  destroy and clean resources
    CUDA_CHECK_RETURN (cudaDeviceReset ());

    return 0;
}
