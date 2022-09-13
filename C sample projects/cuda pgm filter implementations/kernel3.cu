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
#define COMPUTE_UNIT (36*2)
#define THREAD_AMOUNT (256)
#define WARPTHREADS (1024)

__global__ void warpReduceMin3(int32_t *val, int N, int32_t *output)
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

__global__ void warpReduceMax3(int32_t *val, int N, int32_t *output)
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



int32_t findmax3(int N, int32_t *gpumem){
    int blocks = N/WARPTHREADS;
    blocks++;
    if(blocks>1){
        int32_t *max_out;
        cudaMalloc(&max_out, N*sizeof(int32_t));
        warpReduceMax3<<<blocks,WARPTHREADS>>>(gpumem, N, max_out);
        int32_t m = findmax3(blocks, max_out);
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

int32_t findmin3(int N, int32_t *gpumem){
    int blocks = N/WARPTHREADS;
    blocks++;
    if(blocks>1){
        int32_t *min_out;
        cudaMalloc(&min_out, N*sizeof(int32_t));
        warpReduceMin3<<<blocks,WARPTHREADS>>>(gpumem, N, min_out);
        int32_t m = findmin3(blocks, min_out);
        cudaFree(min_out);
        return m;
    }
    else{
        int32_t m = 2147483647;
        int32_t *hostmem;
        hostmem = (int32_t*)malloc(sizeof(int32_t) * N);
        cudaMemcpy(hostmem, gpumem, sizeof(int) * N,  cudaMemcpyDeviceToHost);
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


__device__ int32_t apply2d3(const int8_t *f, int32_t dimension, const int32_t *original, int32_t width, int32_t height, int row, int column) {
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

void print_run3(float time_cpu, int kernel, float time_gpu_computation, float time_gpu_transfer_in, float time_gpu_transfer_out) {
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu / time_gpu_computation);
    printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                            time_gpu_transfer_out));
}


void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, double cputime) 
{
    // init
    int32_t *gpu_in;
    int32_t *gpu_out;
    int8_t *gpu_filter;
    int32_t *buf_min;
    int32_t *buf_max;
    float transfer_in, computation_time, transfer_out; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // malloc 
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
    kernel3<<<COMPUTE_UNIT,THREAD_AMOUNT>>>(gpu_filter, dimension, gpu_in, gpu_out, width, height); 

    cudaMemcpy(buf_max, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf_min, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToDevice);
    // runs with atmoic instructions 
    int warplargest = findmax3(width*height, buf_max);
    int warpsmallest = findmin3(width*height, buf_min);

    if (warpsmallest != warplargest){
        normalize3<<<COMPUTE_UNIT,THREAD_AMOUNT>>>(gpu_out, width, height, warpsmallest, warplargest);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    cudaEventRecord(start);
    
    cudaMemcpy(output, gpu_out, width*height*sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_out, start, stop);
    print_run3(cputime, 3, computation_time, transfer_in, transfer_out);
    cudaFree(gpu_in);
    cudaFree(gpu_out);
    cudaFree(gpu_filter);
    cudaFree(buf_max);
    cudaFree(buf_min);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
}

__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height) 
                        
{
    int id = (threadIdx.x  + (blockIdx.x * blockDim.x) ) ;
    int workload = (width * height) / (COMPUTE_UNIT * blockDim.x);
    int location = workload *  id;
    int end = workload * (id+1);
    if (blockIdx.x == COMPUTE_UNIT -1 && threadIdx.x == THREAD_AMOUNT-1){
        end = (width * height);
    }
    
    while (location < width * height && location < end){
        output[location] = apply2d3(filter, dimension, input, width, height, location/width, location%width);
        location++;
    }
    
    __syncthreads();

}


__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) 
{
    int id = (threadIdx.x  + (blockIdx.x * blockDim.x) ) ;
    int workload = (width * height) / (COMPUTE_UNIT * blockDim.x);
    int location = workload * id;
    int end = workload * (id+1);
    if (blockIdx.x == COMPUTE_UNIT -1 && threadIdx.x == THREAD_AMOUNT-1){
        end = (width * height);
    }
    while (location < width * height && location < end){
        if (width * height == 1){
            printf("before %d %d %d %d\n", image[location],(image[location] - smallest) * 255, biggest - smallest, (image[location] - smallest) * 255/ biggest - smallest );
        }
        image[location] = ((image[location] - smallest) * 255) / (biggest - smallest);
        if (width * height == 1){
            printf("after %d\n", image[location]);
        }
        location++;
    }
    __syncthreads();
   
}
