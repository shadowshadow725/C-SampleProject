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

#define WARPTHREADS (1024)

__device__ int32_t apply2d(const int8_t *f, int32_t dimension, const int32_t *original, int32_t width, int32_t height, int row, int column) {
    int size = width * height;
    int32_t pixel = 0;
    row -= dimension/2;
    column-= dimension/2;
    // loop to add all the pixels 
    for(int i = 0;i<dimension;i++){
        for (int j = 0;j<dimension;j++){
            int location = row*width + column;
            if (0 <= location && location < size && row > -1 && column > -1 && row < height && column < width){
                pixel += original[location] * f[i * dimension + j];
            }
            column++;
            
        }
        row++;
        column-=dimension;
    }
    
    return pixel;
} 


void print_run2(float time_cpu, int kernel, float time_gpu_computation, float time_gpu_transfer_in, float time_gpu_transfer_out) {
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu / time_gpu_computation);
    printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                            time_gpu_transfer_out));
}

__global__ void warpReduceMin(int32_t *val, int N, int32_t *output)
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

__global__ void warpReduceMax(int32_t *val, int N, int32_t *output)
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

int32_t findmax(int N, int32_t *gpumem){
    int blocks = N/WARPTHREADS;
    blocks++;
    if(blocks>1){
        int32_t *max_out;
        cudaMalloc(&max_out, N*sizeof(int32_t));
        warpReduceMax<<<blocks,WARPTHREADS>>>(gpumem, N, max_out);
        int32_t m = findmax(blocks, max_out);
        cudaFree(max_out);
        return m;
    }
    else{
        int32_t m = -2147483647;
        int32_t *hostmem;
        hostmem = (int32_t*)malloc(sizeof(int32_t) * N);
        cudaMemcpy(hostmem, gpumem, sizeof(int32_t) * N,  cudaMemcpyDeviceToHost);
        for(int i = 0;i<N;i++){
            if(hostmem[i] > m){
                m = hostmem[i];
            }
        }
        free(hostmem);
        return m;
    }
    return -1;
}

int32_t findmin(int N, int32_t *gpumem){
    int blocks = N/WARPTHREADS;
    blocks++;
    if(blocks>1){
        int32_t *min_out;
        cudaMalloc(&min_out, N*sizeof(int32_t));
        warpReduceMin<<<blocks,WARPTHREADS>>>(gpumem, N, min_out);
        int32_t m = findmin(blocks, min_out);
        cudaFree(min_out);
        return m;
    }
    else{
        int32_t m = 2147483647;
        int32_t *hostmem;
        hostmem = (int32_t*)malloc(sizeof(int32_t) * N);
        cudaMemcpy(hostmem, gpumem, sizeof(int32_t) * N,  cudaMemcpyDeviceToHost);
        for(int i = 0;i<N;i++){
            if(hostmem[i] < m){
                m = hostmem[i];
            }
        }
        free(hostmem);
        return m;
    }
    return -1;
}


void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input, int32_t *output, int32_t width, int32_t height, double cputime) {
    // Figure out how to split the work into threads and call the kernel below.
    // init
    int32_t *gpu_in;
    int32_t *gpu_out;
    int32_t *buf_min;
    int32_t *buf_max;
    int8_t *gpu_filter;
    int blockcount = width*height/1024;
    blockcount++;
    float transfer_in, computation_time, transfer_out; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc(&gpu_filter, dimension*dimension*sizeof(int8_t));
    cudaMalloc(&gpu_in, width*height*sizeof(int32_t));
    cudaMalloc(&gpu_out, width*height*sizeof(int32_t));
    cudaMalloc(&buf_min, width*height*sizeof(int32_t));
    cudaMalloc(&buf_max, width*height*sizeof(int32_t));
    cudaMemcpy(gpu_in, input, width*height*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_filter, filter, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_in, start, stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // run kernel 
    kernel2<<<blockcount,1024>>>(gpu_filter, dimension, gpu_in, gpu_out, width, height);    
    
    cudaMemcpy(buf_max, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf_min, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToDevice);
    // runs with atmoic instructions 
    int warplargest = findmax(width*height, buf_max);
    int warpsmallest = findmin(width*height, buf_min);


    if (warpsmallest != warplargest){
        normalize2<<<blockcount,1024>>>(gpu_out, width, height, warpsmallest, warplargest);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);

    cudaEventRecord(start);

    cudaMemcpy(output, gpu_out, width * height * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_out, start, stop);
    print_run2(cputime, 2, computation_time, transfer_in, transfer_out);
    // cleanup 
    cudaFree(buf_max);
    cudaFree(buf_min);
    cudaFree(gpu_in);
    cudaFree(gpu_out);
    cudaFree(gpu_filter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void kernel2(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height) 
{
    int location = (threadIdx.x  + (blockIdx.x * blockDim.x) ) ;
    if ( location < width * height){
        output[location] = apply2d(filter, dimension, input, width, height, location / width, location % width);
    }
    __syncthreads();

}

__global__ void normalize2(int32_t *image, int32_t width, int32_t height,
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
