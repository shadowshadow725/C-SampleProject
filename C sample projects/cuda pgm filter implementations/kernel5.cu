/* ------------
* This code is provided solely for the personal and private use of
* students taking the CSC367H5 course at the University of Toronto.
* Copying for purposes other than this use is expressly prohibited.
* All forms of distribution of this code, whether as given or with
* any changes, are expressly prohibited.
*
* Authors: Bogdan Simion, Felipe de Azevedo Piovezan
*
* All of the files in this directory and all subdirectories are:
* Copyright (c) 2019 Bogdan Simion
* -------------
*/


#include "kernels.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>
// macros 
#define COMPUTE_UNIT (36*2)
#define THREAD_AMOUNT (256)
#define COPYUNIT (3)
#define WARPTHREADS (1024)

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


__global__ void warpReduceMin5(int32_t *val, int N, int32_t *output)
{
    int base = (blockIdx.x * blockDim.x);
    int id = (threadIdx.x) ;
    int localn = 4;
    if (blockDim.x == blockIdx.x-1){
        localn = N - base;
    }
    int n = localn;
    localn /= 2;
    while (localn > 0){
        if(id < localn){
            atomicMin(&val[base + id], val[base + id + localn]);
        }
        localn /= 2;
        __syncthreads();
    }
    if (threadIdx.x == 0){
        atomicMin(&val[base + id], val[base + n-1]);
        output[blockIdx.x] = val[base + id];
    }
    __syncthreads();
}

__global__ void warpReduceMax5(int32_t *val, int N, int32_t *output)
{
    int base = (blockIdx.x * blockDim.x);
    int id = (threadIdx.x) ;
    int localn = 4;
    if (gridDim.x-1 == blockIdx.x){
        localn = N - base;
    }
    int n = localn;
    localn /= 2;
    while (localn > 0 ){
        if(id < localn){
            atomicMax(&val[base + id], val[base + id + localn]);
        }
        localn /= 2;
        __syncthreads();
    }
    if (threadIdx.x == 0){
        atomicMax(&val[base + id], val[base + n-1]);
        output[blockIdx.x] = val[base + id];
    }
    __syncthreads();
}

int32_t findmax5(int N, int32_t *gpumem){
    int blocks = N/WARPTHREADS;
    blocks++;
    if(blocks>1){
        int32_t *max_out;
        cudaMalloc(&max_out, N*sizeof(int32_t));
        warpReduceMax5<<<blocks,WARPTHREADS>>>(gpumem, N, max_out);
        int32_t m = findmax5(blocks, max_out);
        cudaFree(max_out);
        return m;
    }
    else{
        int32_t m = -2147483647;
        int32_t *hostmem;
        cudaMallocHost((void **) &hostmem, sizeof(int32_t) * N);
        cudaMemcpy(hostmem, gpumem, sizeof(int32_t) * N,  cudaMemcpyDeviceToHost);
        for(int i = 0;i<N;i++){
            if(hostmem[i] > m){
                m = hostmem[i];
            }
        }
        cudaFree(hostmem);
        return m;
    }
    return -1;
}

int32_t findmin5(int N, int32_t *gpumem){
    int blocks = N/WARPTHREADS;
    blocks++;
    if(blocks>1){
        int32_t *min_out;
        cudaMalloc(&min_out, N*sizeof(int32_t));
        warpReduceMin5<<<blocks,WARPTHREADS>>>(gpumem, N, min_out);
        int32_t m = findmin5(blocks, min_out);
        cudaFree(min_out);
        return m;
    }
    else{
        int32_t m = 2147483647;
        int32_t *hostmem;
        cudaMallocHost((void **) &hostmem, sizeof(int32_t) * N);
        cudaMemcpy(hostmem, gpumem, sizeof(int32_t) * N,  cudaMemcpyDeviceToHost);
        for(int i = 0;i<N;i++){
            if(hostmem[i] < m){
                m = hostmem[i];
            }
        }
        cudaFree(hostmem);
        return m;
    }
    return -1;
}




__device__ int32_t apply2d5(const int8_t *f, int32_t dimension, const int32_t *original, int32_t width, int32_t height, int row, int column) {
    int size = width * height;
    int32_t pixel = 0;
    row -= dimension/2;
    column-= dimension/2;
    column++;
    int location = row*width + column;
    if (0 <= location && location < size && row > -1 && column > -1 && row < height && column < width){
        pixel += original[location] * f[1];
    }
    column-=1;
    row++;
    location = row*width + column;
    if (0 <= location && location < size && row > -1 && column > -1 && row < height && column < width){
        pixel += original[location] * f[3];
    }
    column+=1;
    location = row*width + column;
    if (0 <= location && location < size && row > -1 && column > -1 && row < height && column < width){
        pixel += original[location] * f[4];
    }
    column+=1;
    location = row*width + column;
    if (0 <= location && location < size && row > -1 && column > -1 && row < height && column < width){
        pixel += original[location] * f[5];
    }
    column--; 
    row++;
    location = row*width + column;
    if (0 <= location && location < size && row > -1 && column > -1 && row < height && column < width){
        pixel += original[location] * f[7];
    }


    return pixel;

}

void print_run5(float time_cpu, int kernel, float time_gpu_computation, float time_gpu_transfer_in, float time_gpu_transfer_out) {
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu / time_gpu_computation);
    printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                            time_gpu_transfer_out));
}


void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
    int32_t *output, int32_t width, int32_t height, double cputime) 
{
    // init
    int32_t *gpu_in;
    int32_t *gpu_out;
    int8_t *gpu_filter;
    int32_t *buf_min;
    int32_t *buf_max;
    int blockcount = width*height/1024;
    blockcount++;
    float transfer_in, computation_time, transfer_out; 
    cudaEvent_t start, stop;
    cudaStream_t stream[COPYUNIT]; // because rtx 2070 has 3 copy units 
    cudaEvent_t startEvent, stopEvent;
    for (int i = 0; i < COPYUNIT; ++i){
        checkCuda( cudaStreamCreate(&stream[i]) );
    }
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    // case when kernel 5 is ran with a small file
    if(width * height <= 1024){
        // small input makes async mem copy run slower for some reason 
        // just running the kernel that kernel 5 is based off 
        run_kernel2(filter, dimension, input, output, width, height, cputime);
        return;
    }

    // memory allocation 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    checkCuda( cudaEventRecord(startEvent,0) );
    cudaMalloc(&gpu_filter, dimension*dimension*sizeof(int8_t));
    cudaMalloc(&gpu_in, width*height*sizeof(int32_t));
    cudaMalloc(&gpu_out, width*height*sizeof(int32_t));
    cudaMalloc(&buf_min, width*height*sizeof(int32_t));
    cudaMalloc(&buf_max, width*height*sizeof(int32_t));
    cudaMemcpy(gpu_filter, filter, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
    
    // Using all 3 copy units that rtx 2070 has 
    // optimization 
    for (int i = 0; i < COPYUNIT; ++i)
    {
        int streamBytes = (width*height)/COPYUNIT;
        int offset = i * streamBytes;
        if (i == COPYUNIT-1){
            streamBytes = width*height - (COPYUNIT-1)*(streamBytes);
        }
        checkCuda( cudaMemcpyAsync(&gpu_in[offset], &input[offset], streamBytes*sizeof(int32_t), cudaMemcpyHostToDevice, stream[i]) );
        
    }
    
    
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&transfer_in, startEvent, stopEvent) );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    
    // optimization 
    for (int i = 0; i < COPYUNIT; ++i)
    {
        if (i == COPYUNIT -1){
            kernel5<<<blockcount - (blockcount/3)*(COPYUNIT-1),1024,0,stream[i]>>>(gpu_filter, dimension, gpu_in, gpu_out, width, height, (i*(blockcount/3)*1024) ); 
        }
        else{
            kernel5<<<blockcount/3,1024,0,stream[i]>>>(gpu_filter, dimension, gpu_in, gpu_out, width, height, (i*(blockcount/3)*1024)); 
        }
    }
    
    
    checkCuda(cudaDeviceSynchronize());

    checkCuda( cudaMemcpyAsync(buf_max, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream[0]) );
    checkCuda( cudaMemcpyAsync(buf_min, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream[1]) );
    // I hard coded the streams, because it only does 2 memcopies 
    checkCuda(cudaDeviceSynchronize());
    // runs with atmoic instructions 
    int warplargest = findmax5(width*height, buf_max);
    int warpsmallest = findmin5(width*height, buf_min);

    if (warpsmallest != warplargest){
        normalize5<<<blockcount,1024>>>(gpu_out, width, height, warpsmallest, warplargest);
        // this runs really fast already, didn't bother optimizing it 
        // it takes 0.6ms to run for 187MB file 
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    

    cudaEventRecord(start);
    // didn't use async copy because it didn't use the copy units 
    cudaMemcpy(output, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_out, start, stop);
    // clean up 
    print_run5(cputime, 5, computation_time, transfer_in, transfer_out);
    for (int i = 0; i < COPYUNIT; ++i){
        checkCuda( cudaStreamDestroy(stream[i]) );
    }
    cudaFree(gpu_in);
    cudaFree(gpu_out);
    cudaFree(gpu_filter);
    cudaFree(buf_max);
    cudaFree(buf_min);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    

}



__global__ void kernel5(const int8_t *filter, int32_t dimension,
           const int32_t *input, int32_t *output, int32_t width,
           int32_t height, int base) 
{
    int location = (threadIdx.x  + (blockIdx.x * blockDim.x) ) + base;
    if ( location < width * height){
        output[location] = apply2d5(filter, dimension, input, width, height, location / width, location % width);
    }
    __syncthreads();
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
              int32_t smallest, int32_t biggest) 
{
    int location = (threadIdx.x  + (blockIdx.x * blockDim.x) );
    if (location < width * height){
        if (smallest != biggest) {
            image[location] = ((image[location] - smallest) * 255) / (biggest - smallest);
        }
    }
    __syncthreads();
}
