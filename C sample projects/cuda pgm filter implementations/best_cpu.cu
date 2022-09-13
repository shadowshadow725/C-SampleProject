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
#include <stdio.h>
#include "kernels.h"
#include <pthread.h>
#include <sys/sysinfo.h>
#define CPU_THREAD (8)


int32_t largest = -2147483647;
int32_t smallest = 2147483647;
pthread_mutex_t mutex;
typedef struct filter_t {
    int32_t dimension;
    const int8_t *matrix;
  } filter;
typedef struct common_work_t
{
    const filter *f;
    const int32_t *original_image;
    int32_t *output_image;
    int32_t width;
    int32_t height;
    int32_t max_threads;
    pthread_barrier_t barrier;
} common_work;
typedef struct work_t
{
      common_work *common;
      int32_t id;
} work;

int32_t apply2d(const filter *f, const int32_t *original, int32_t *target, int32_t width, int32_t height, int row, int column) {
    
    int size = width * height;
    int32_t pixel = 0;
    int32_t n = 0;
    row -= f->dimension/2;
    column-= f->dimension/2;
    // loop to add all the pixels 
    for(int i = 0;i<f->dimension;i++){
        for (int j = 0;j<f->dimension;j++){
            int location = row*width + column;
            if (0 <= location && location < size && row > -1 && column > -1 && row < height && column < width){
                pixel += original[location] * f->matrix[i * f->dimension + j];
                n++;
            }
            column++;
            
        }
        row++;
        column-=f->dimension;
    }
    
    return pixel;

}
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest, int32_t largest) 
{
    if (smallest == largest) {
        return;
    }
    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

void *horizontal_sharding (void *args) {
    work w = *((work*)args);
    // make a copy of the data on the local stack 
    // to avoid accessing shared data 
    int height = w.common->height;
    int width = w.common->width;
    int id = w.id;
    // determine bounds 
    int start = id * (w.common->height/w.common->max_threads);
    int end = (id +1) * (w.common->height/w.common->max_threads);
    if (id == w.common->max_threads -1){
        end = w.common->height;
    }
    const filter *f = w.common->f;
    const int32_t *original = w.common->original_image;
    int32_t *target = w.common->output_image;

    int32_t l = -2147483646;
    int32_t s = 2147483647;
    // using apply2d to calculate pixels 
    for (int i = start; i < end; i++){
        for( int j =0;j < width;j++){
            target[i*width + j] = apply2d(f, original, target, width, height, i, j);
            // save largest and smallest for normalize 
            if (target[i*width + j] > l){
                l = target[i*width + j];
            }
            if (target[i*width + j] < s){
                s = target[i*width + j];
            }
        }
    }
    // wait for all threads to finish calculating 
    pthread_barrier_wait(&(w.common->barrier));
    // then update the global largest value and smallest value 
    pthread_mutex_lock(&mutex);
    if (l > largest){
        largest = l;
    }
    if (s < smallest){
        smallest = s;
    }
    
    pthread_mutex_unlock(&mutex);
    pthread_barrier_wait(&(w.common->barrier));
    
    // start normalize process 
    for (int i = start; i < end; i++){
        for( int j =0;j < width;j++){
            normalize_pixel(target, i*width + j, smallest, largest);
        }
    }
    
    return NULL;
    
}


void run_best_cpu(const int8_t *filterdat, int32_t dimension, const int32_t *input, int32_t *output, int32_t width, int32_t height) {
    // this is for when best cpu is ran multiple times in the same program because
    // my implementation will change globals
    // this is required for tests.cu to pass 
    largest = -2147483647;
    smallest = 2147483647;
    int num_threads = CPU_THREAD;
    // initialize common work and the work array 
    common_work *cw = (common_work*)malloc(sizeof(struct common_work_t));
    work **threads_work = (work**)malloc(sizeof(work*) * num_threads);
    const filter lp3_f = {dimension, filterdat};
    cw->f = &lp3_f;
    cw->height = height;
    cw->width = width;
    cw->original_image = input;
    cw->output_image = output;
    cw->max_threads = num_threads;
    // init the barrier to sync threads 
    pthread_barrier_init(&(cw->barrier) ,NULL, num_threads);
    
    pthread_t thread [num_threads];
    // linking common work with each work piece 
    for (int i = 0;i< num_threads;i++){
        threads_work[i] = (work*) malloc(sizeof(work));
        threads_work[i]->common = cw;
        threads_work[i]->id = i;
    }

    // pin cpu blob 
    int nthreads = get_nprocs();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (int i = 0;i< num_threads;i++){
        pthread_create(&thread[i], NULL, horizontal_sharding , (void*)(threads_work[i]));
        CPU_SET(i%nthreads, &cpuset);
        pthread_setaffinity_np(thread[i], sizeof(cpu_set_t), &cpuset);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread[i], NULL);
    }
    for (int i = 0; i < num_threads; i++) {
        free(threads_work[i]);
    }
    free(cw);
    free(threads_work);
}
