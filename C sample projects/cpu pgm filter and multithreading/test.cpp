
#include "filters.h"
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include "time_util.h"
#include <math.h>
#include <sys/sysinfo.h>
#include <unistd.h> 
#include <string.h>

#define TESTSIZE 4000
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



/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] = {
    0, 1, 0, 1, -4, 1, 0, 1, 0,
};
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
filter lp5_f = {5, lp5_m};

/* Laplacian of gaussian */
int8_t log_m[] = {
    0, 1, 1, 2, 2, 2,   1,   1,   0, 1, 2, 4, 5, 5,   5,   4,   2,
    1, 1, 4, 5, 3, 0,   3,   5,   4, 1, 2, 5, 3, -12, -24, -12, 3,
    5, 2, 2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3,   -12, -24, -12,
    3, 5, 2, 1, 4, 5,   3,   0,   3, 5, 4, 1, 1, 2,   4,   5,   5,
    5, 4, 2, 1, 0, 1,   1,   2,   2, 2, 1, 1, 0,
};
filter log_f = {9, log_m};

/* Identity */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};
filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

typedef struct Queue { 
    Queue *next = NULL;
    int32_t row_start;
    int32_t row_end;
    int32_t colume_start;
    int32_t colume_end;
}queue; 



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

// globals 
pthread_mutex_t mutex;
int32_t largest = -2147483647;
int32_t smallest = 2147483647;
queue *q = (queue*)malloc(sizeof(queue));
queue *normalize_queue = q;
queue *dallocq = q;
int qleng;

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) {
  if (smallest == largest) {
    return;
  }

  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}
/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
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


void apply_filter2d(const filter *f, const int32_t *original, int32_t *target, int32_t width, int32_t height) {
    int32_t l = -2147483647;
    int32_t s = 2147483647;
    // loop through the whole array using apply2d to calculate each pixel 
    for (int i = 0; i < height;i++){
        for (int j = 0; j < width; j++){
            target[i*width + j] = apply2d(f, original, target, width, height, i, j);
            // save the largest and smallest values for normalizing 
            if (target[i*width + j] < s ){
                s = target[i*width + j];

            }
            if (target[i*width + j] > l){
                l =target[i*width + j];
            }
        }
        
    }
    // normalize loop 
    for (int i = 0; i < height;i++){
        for (int j = 0; j < width; j++){
            normalize_pixel(target, i*width + j, s, l);
        }

    }
    
    
}


// free work queue function 
void work_chunk_free(){
    if (qleng < 0){
        return;
    }

    for ( int i = 0;i<qleng;i++){
        queue *qq = dallocq;
        dallocq = dallocq->next;
        free(qq);
    }

    q = (queue*)malloc(sizeof(queue));
    q->next = NULL;
    normalize_queue = q;
    dallocq = q;
    qleng = 1;
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

    int32_t l = -2147483647;
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

void *vertical_sharding_column_major(void *args){
    work *w = ((work*)args);
    // make a copy of the data on the local stack 
    // to avoid accessing shared data 
    int height = w->common->height;
    int width = w->common->width;
    int id = w->id;
    // determine bounds 
    int start = id * (width/w->common->max_threads);
    int end = (id +1) * (width/w->common->max_threads);
    if (id == w->common->max_threads -1){
        end = width;
    }
    const filter *f = w->common->f;
    const int32_t *original = w->common->original_image;
    int32_t *target = w->common->output_image;

    int32_t l = -2147483647;
    int32_t s = 2147483647;
    // using apply2d to calculate pixels 
    for (int col = start;col<end;col++){
        for (int row = 0; row < height; row++){
            target[row*width + col] = apply2d(f, original, target, width, height, row, col);
            // save largest and smallest for normalize 
            if (target[row*width + col]  > l){
                l = target[row*width + col] ;
            }
            if (target[row*width + col]  < s){
                s = target[row*width + col] ;
            }
        }
    }
    // wait for all threads to finish calculating 
    pthread_barrier_wait(&(w->common->barrier));
    // then update the global largest value and smallest value 
    pthread_mutex_lock(&mutex);
    if (l > largest){
        largest = l;

    }
    if (s < smallest){
        smallest = s;
    }
    pthread_mutex_unlock(&mutex);
    pthread_barrier_wait(&(w->common->barrier));
    // start normalize process 
    for (int col = start;col<end;col++){
        for (int row = 0; row < height; row++){
            normalize_pixel(target, row*width + col, smallest, largest);            
        }
    }
    
    
    return NULL;

}   


void *vertical_sharding_row_major(void *args){
    work *w = ((work*)args);
    // make a copy of the data on the local stack 
    // to avoid accessing shared data 
    int height = w->common->height;
    int width = w->common->width;
    int id = w->id;
    // set bounds 
    int start = id * (width/w->common->max_threads);
    int end = (id +1) * (width/w->common->max_threads);
    if (id == w->common->max_threads -1){
        end = width;
    }
    const filter *f = w->common->f;
    const int32_t *original = w->common->original_image;
    int32_t *target = w->common->output_image;

    int32_t l = -2147483647;
    int32_t s = 2147483647;
    // calculate pixels using apply2d
    for (int row = 0; row < height; row++){
        for (int col = start;col<end;col++){
            target[row*width + col] = apply2d(f, original, target, width, height, row, col);
            // save largest and smallest for normailization 
            if (target[row*width + col]  > l){
                l = target[row*width + col] ;
            }
            if (target[row*width + col]  < s){
                s = target[row*width + col] ;
            }
        }
        
    }
    // wait for all threads to finish calculating pixels 
    pthread_barrier_wait(&(w->common->barrier));
    // update global max and min 
    pthread_mutex_lock(&mutex);
    if (l > largest){
        largest = l;

    }
    if (s < smallest){
        smallest = s;
    }
    pthread_mutex_unlock(&mutex);
    pthread_barrier_wait(&(w->common->barrier));
    // start normailization process 
    for (int col = start;col<end;col++){
        for (int row = 0; row < height; row++){
            normalize_pixel(target, row*width + col, smallest, largest);            
        }
    }
    return NULL;


}

// not complete 
// multi threading bugs 



int lazyfix;
void *work_chunk_func(void *args){
    work *w = ((work*)args);
    int height = w->common->height;
    int width = w->common->width;
    int id = w->id;
    const filter *f = w->common->f;
    const int32_t *original = w->common->original_image;
    int32_t *target = w->common->output_image;
    int32_t l = -2147483647;
    int32_t s = 2147483647;
  
    

    pthread_mutex_lock(&mutex);
    while(q->next != NULL && lazyfix > 1){
        
        
        if (q->next!= NULL && lazyfix > 1){
            queue *bounds = q;
            q = q->next;
            lazyfix--;
            
            int32_t row_start = bounds->row_start;
            int32_t row_end = bounds->row_end;
            int32_t colume_start = bounds->colume_start;
            int32_t colume_end = bounds->colume_end;
            pthread_mutex_unlock(&mutex);
            if (row_end > height){
                row_end = height;
            }
            if (colume_end > width){
                colume_end = width;
            }
            if (row_start >= height){
                row_start = row_end;
            }
            if (colume_start >= colume_end){
                colume_start = colume_end;
            }
            if(row_start < 0){
                row_start = 0;
            }
            if (colume_start < 0){
                colume_start = 0;
            }
            for ( int row = row_start; row < row_end;row++){
                for ( int col = colume_start; col< colume_end; col++){
                    target[row*width + col] = apply2d(f, original, target, width, height, row, col);
                    
                    if (target[row*width + col]  > l){
                        l = target[row*width + col] ;
                    }
                    if (target[row*width + col]  < s){
                        s = target[row*width + col] ;
                    }
                }
            }
        }
        else
        {
            pthread_mutex_unlock(&mutex);
            
        }
        pthread_mutex_lock(&mutex);

    }
    pthread_mutex_unlock(&mutex);
    pthread_barrier_wait(&(w->common->barrier));
    pthread_mutex_lock(&mutex);
    if (l > largest){
        largest = l;

    }
    if (s < smallest){
        smallest = s;
    }
    lazyfix = qleng;
    pthread_mutex_unlock(&mutex);

    pthread_barrier_wait(&(w->common->barrier));
    
    pthread_mutex_lock(&mutex);
    while(normalize_queue->next != NULL && lazyfix > 1){
        
        
        if (normalize_queue->next != NULL && lazyfix > 1){
            lazyfix--;
            queue *bounds = normalize_queue;
            normalize_queue = normalize_queue->next;
            pthread_mutex_unlock(&mutex);
            int32_t row_start = bounds->row_start;
            int32_t row_end = bounds->row_end;
            int32_t colume_start = bounds->colume_start;
            int32_t colume_end = bounds->colume_end;
            if (row_end >= height){
                row_end = height;
            }
            if (colume_end >= width){
                colume_end = width;
            }
            if (row_start >= height){
                row_start = row_end;
            }
            if (colume_start >= colume_end){
                colume_start = colume_end;
            }
            if(row_start < 0){
                row_start = 0;
            }
            if (colume_start < 0){
                colume_start = 0;
            }
            
            for ( int row = row_start; row < row_end;row++){
                for ( int col = colume_start; col< colume_end; col++){
                    normalize_pixel(target, row*width + col, smallest, largest);
                }
            }
        }
        else
        {
            pthread_mutex_unlock(&mutex);
            
        }
        pthread_mutex_lock(&mutex);
    }
    pthread_mutex_unlock(&mutex);
    return NULL;

}



void apply_filter2d_threaded(const filter *f, const int32_t *original,
                             int32_t *target, int32_t width, int32_t height,
                             int32_t num_threads, parallel_method method,
                             int32_t work_chunk) {
                        
    // initialize common work and the work array 
    common_work *cw = (common_work*)malloc(sizeof(struct common_work_t));
    work **threads_work = (work**)malloc(sizeof(work*) * num_threads);

    cw->f = f;
    cw->height = height;
    cw->width = width;
    cw->original_image = original;
    cw->output_image = target;
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




    // check method and start threads 
    // also pin it to a cpu 
    if (method == SHARDED_ROWS){
        for (int i = 0;i< num_threads;i++){
            pthread_create(&thread[i], NULL, horizontal_sharding , (void*)(threads_work[i]));
            CPU_SET(i%nthreads, &cpuset);
            pthread_setaffinity_np(thread[i], sizeof(cpu_set_t), &cpuset);
        }
    }
    else if (method == SHARDED_COLUMNS_COLUMN_MAJOR){
        for (int i = 0;i< num_threads;i++){
            pthread_create(&thread[i], NULL, vertical_sharding_column_major, (void*)(threads_work[i]));
            CPU_SET(i%nthreads, &cpuset);
            pthread_setaffinity_np(thread[i], sizeof(cpu_set_t), &cpuset);
        }
    }
    else if (method == SHARDED_COLUMNS_ROW_MAJOR){
        for (int i = 0;i< num_threads;i++){
            pthread_create(&thread[i], NULL, vertical_sharding_row_major, (void*)(threads_work[i]));
            CPU_SET(i%nthreads, &cpuset);
            pthread_setaffinity_np(thread[i], sizeof(cpu_set_t), &cpuset);
        }
    }
    else if (method == WORK_QUEUE){
        if (work_chunk >= width && work_chunk >= height){
            apply_filter2d_threaded(f, original, target, width, height, num_threads, SHARDED_ROWS, 1);
            return;
        }
        else
        {
            queue *head = q;
            int qlen = 1;
            for (int i = 0;i < ((height/work_chunk)+1); i++){
                for ( int j = 0; j < ((width/work_chunk)+1);j++){
                    q->row_start = work_chunk * i;
                    q->row_end = work_chunk * (i+1);
                    if (work_chunk > height){
                        q->row_end = height;
                    }
                    q->colume_start = work_chunk * j;
                    q->colume_end = work_chunk * (j+1);
                    if(work_chunk > width){
                        q->colume_end = width;
                    }
                    
                    q->next = (queue*)malloc(sizeof(queue));
                    q = q->next;
                    qlen++;
                }
            }
            lazyfix = qlen;
            qleng = qlen;
            q = head;
            for (int i = 0;i< num_threads;i++){
                pthread_create(&thread[i], NULL, work_chunk_func, (void*)(threads_work[i]));
                CPU_SET(i%nthreads, &cpuset);
                pthread_setaffinity_np(thread[i], sizeof(cpu_set_t), &cpuset);
            }
            for (int i = 0; i < num_threads; i++) {
                pthread_join(thread[i], NULL);
            }
        }
        
        
        
    }
    // wait for all threads to finish 
    if (method != WORK_QUEUE){
        for (int i = 0; i < num_threads; i++) {
            pthread_join(thread[i], NULL);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        free(threads_work[i]);
    }
    free(cw);
    free(threads_work);
    largest = -2147483647;
    smallest = 2147483647;
    // is work queue is used free it 
    if (method == WORK_QUEUE){
        work_chunk_free();
    }
}




int32_t image[] = {1, 4, 7, 10, 3, 
                   2, 5, 8, 1, 4, 
                   3, 6, 9, 2, 5};

int8_t tf_m[] = {
    0, 1, 0,
     1, -4, 1, 
     0, 1, 0
};
filter tf_f = {3, tf_m};
int32_t matrix[TESTSIZE * TESTSIZE];
int32_t target3[TESTSIZE * TESTSIZE];
static double ranf(void)
{
	return (int)random() % 255;
}

void generate(int m[TESTSIZE * TESTSIZE])
{
	srandom(42);
	for (int i = 0; i < TESTSIZE * TESTSIZE; i++) {
		m[i] = (int)ranf();
	}
}

int32_t image2[] = {1, 4, 7, 10, 3, 2, 5, 8, 1, 4, 3, 6, 9, 2, 5};
int main (){
    generate(matrix);
    struct timespec start;
	struct timespec end;
	double time_msec = 0.0;
	clockid_t clock = CLOCK_MONOTONIC;	
    FILE *f = fopen("SEQUENTIAL.txt", "w");
	

    clock_gettime(clock, &start);
    apply_filter2d(&tf_f, matrix, target3, TESTSIZE, TESTSIZE);
	clock_gettime(clock, &end);
    time_msec = timespec_to_msec(difftimespec(end, start));
    fprintf(f, "%fms\n", time_msec);

    clock_gettime(clock, &start);
    apply_filter2d(&lp5_f, matrix, target3, TESTSIZE, TESTSIZE);
	clock_gettime(clock, &end);
    time_msec = timespec_to_msec(difftimespec(end, start));
    fprintf(f, "%fms\n", time_msec);
	fclose(f);



	f = fopen("SHARDED_ROWS.txt", "w");
    for( int t = 1;t<9;t++){
        clock_gettime(clock, &start);
        apply_filter2d_threaded(&tf_f, matrix, target3, TESTSIZE, TESTSIZE, t, SHARDED_ROWS, 1);
        clock_gettime(clock, &end);
        time_msec = timespec_to_msec(difftimespec(end, start));
        fprintf(f, "%fms %d\n", time_msec, t);
        
    }
    fprintf(f, "\n");
    for( int t = 1;t<9;t++){
        clock_gettime(clock, &start);
        apply_filter2d_threaded(&lp5_f, matrix, target3, TESTSIZE, TESTSIZE, t, SHARDED_ROWS, 1);
        clock_gettime(clock, &end);
        time_msec = timespec_to_msec(difftimespec(end, start));
        fprintf(f, "%fms %d\n", time_msec, t);
        
    }
    fclose(f);

    f = fopen("SHARDED_COLUMNS_COLUMN_MAJOR.txt", "w");
    for( int t = 1;t<9;t++){
        clock_gettime(clock, &start);
        apply_filter2d_threaded(&tf_f, matrix, target3, TESTSIZE, TESTSIZE, t, SHARDED_COLUMNS_COLUMN_MAJOR, 1);
        clock_gettime(clock, &end);
        time_msec = timespec_to_msec(difftimespec(end, start));
        fprintf(f, "%fms %d\n", time_msec, t);
        
    }
    fprintf(f, "\n");
    for( int t = 1;t<9;t++){
        clock_gettime(clock, &start);
        apply_filter2d_threaded(&lp5_f, matrix, target3, TESTSIZE, TESTSIZE, t, SHARDED_COLUMNS_COLUMN_MAJOR, 1);
        clock_gettime(clock, &end);
        time_msec = timespec_to_msec(difftimespec(end, start));
        fprintf(f, "%fms %d\n", time_msec, t);
        
    }
    fclose(f);


    f = fopen("SHARDED_COLUMNS_ROW_MAJOR.txt", "w");
    for( int t = 1;t<9;t++){
        clock_gettime(clock, &start);
        apply_filter2d_threaded(&tf_f, matrix, target3, TESTSIZE, TESTSIZE, t, SHARDED_COLUMNS_ROW_MAJOR, 1);
        clock_gettime(clock, &end);
        time_msec = timespec_to_msec(difftimespec(end, start));
        fprintf(f, "%fms %d\n", time_msec, t);
        
    }
    fprintf(f, "\n");
    for( int t = 1;t<9;t++){
        clock_gettime(clock, &start);
        apply_filter2d_threaded(&lp5_f, matrix, target3, TESTSIZE, TESTSIZE, t, SHARDED_COLUMNS_ROW_MAJOR, 1);
        clock_gettime(clock, &end);
        time_msec = timespec_to_msec(difftimespec(end, start));
        fprintf(f, "%fms %d\n", time_msec, t);
        
    }
    fclose(f);

    int sizes[] = {5, 50, 100, 200, 500};

    f = fopen("WORK_QUEUE.txt", "w");

    for(int j = 0 ;j < 5; j++){
        for( int t = 1;t<9;t++){
            clock_gettime(clock, &start);
            apply_filter2d_threaded(&tf_f, matrix, target3, TESTSIZE, TESTSIZE, t, WORK_QUEUE, sizes[j]);
            clock_gettime(clock, &end);
            time_msec = timespec_to_msec(difftimespec(end, start));
            fprintf(f, "%fms %d %d\n", time_msec, t, sizes[j]);
            q = dallocq;
            normalize_queue = dallocq;
        }
        fprintf(f, "\n");
    }
    
    for(int j = 0 ;j < 5; j++){
        for( int t = 1;t<9;t++){
            
            clock_gettime(clock, &start);
            apply_filter2d_threaded(&lp5_f, matrix, target3, TESTSIZE, TESTSIZE, t, WORK_QUEUE, sizes[j]);
            clock_gettime(clock, &end);
            time_msec = timespec_to_msec(difftimespec(end, start));
            fprintf(f, "%fms %d %d\n", time_msec, t, sizes[j]);
            q = dallocq;
            normalize_queue = dallocq;
        }
        fprintf(f, "\n");
    }

    fclose(f);

    
}