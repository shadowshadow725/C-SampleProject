// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC367H5 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Bogdan Simion, Alexey Khrabrov
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2019 Bogdan Simion
// -------------
//1111100111
//0000000010
#define _GNU_SOURCE

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <math.h>
#include "time_util.h"
#include <string.h>

#define PARTARUNS 10000
#define PARTBRUNS 100000

#define L2BATCH (1000)
#define KB (1024)
#define MB (1048576)

static int tests[] = { 1 * KB, 2 * KB, 4 * KB, 8 * KB, 12 * KB, 16 * KB, 24 * KB, 32 * KB, 64 * KB, 128 * KB, 256 * KB, 512 * KB, 1 * MB, 2 * MB, 4 * MB, 6 * MB, 8 * MB, 12 * MB, 16 * MB, 24 * MB, 32 * MB, 48 * MB, 64 * MB};

void parta(float *data, int t){
	struct timespec start;
	struct timespec end;
	double time_msec = 0.0;
	clockid_t clock = CLOCK_MONOTONIC;	
	char *mem1 ;
	char *mem2; 
	char *mem3;
	char *mem4;
	clock_gettime(clock, &start);
	

	clock_gettime(clock, &end);
	time_msec = timespec_to_nsec(difftimespec(end, start));
	int past = 0;
	for (int j = 0 ;j<t ;j++){
		mem1 = malloc (pow(2, 23));
		mem2 = malloc (pow(2, 23));
		mem3 = malloc (pow(2, 23));
		mem4 = malloc (pow(2, 23));
		mem1[0] = 'd';
		mem2[0] = 'd';
		mem3[0] = 'd';
		mem4[0] = 'd';
		for(int i = 1;i<22;i++){
			int present = pow(2, i-1);
			int n = present - past;
			char *s1 = mem1 + past;
			char *s2 = mem2 + past;
			char *s3 = mem3 + past;
			char *s4 = mem4 + past;
			// char *s5 = mem5 + past;

			clock_gettime(clock, &start);
			memset(s1, 'a', n);
			memset(s2, 'a', n);
			memset(s3, 'a', n);
			memset(s4, 'a', n);
			// memset(s5, 'a', n);

			clock_gettime(clock, &end);
			time_msec = timespec_to_nsec(difftimespec(end, start));
			data[i] += time_msec;
			past += n ;

		}
		past = 0;
		free(mem1);
		free(mem2);
		free(mem3);
		free(mem4);
	}


	// for (int i = 0;i<22;i++){
	// 	printf("time = %f ns at index %d \n", data[i]/t*2,(int) pow(2, i));
	// }
	


}

void parta_analysis(float *data, int t ){

	for (int i = 1;i<22; i++){
		if (data[i+1] > data[i]*10){
			double data_perns = pow(2, i+1);
			double time_used = data[i+1]/t;
			double bandwidth = data_perns/time_used * 1000 * 1000 * 1000 / 1024 /1024;
			printf("written %f bytes, in %f ns, effective bandwidth %f MBps\n", data_perns, time_used, bandwidth);
		}
	}
}


void partb_mem_test(int t){
	struct timespec start;
	struct timespec end;
	clockid_t clock = CLOCK_MONOTONIC;	
	char *mem1 ;
	mem1 = malloc (pow(2, 30));
	memset(mem1, 't', pow(2, 30));
	double avg_mem;
	clock_gettime(clock, &start);
	for (int i = 1; i<100000;i++){
		mem1[i*512] = 'a';
	}
	clock_gettime(clock, &end);
	avg_mem = (timespec_to_nsec(difftimespec(end, start))/100000);
	printf("%f ns average latency for system memory\n", avg_mem);
	free(mem1);
	
}



long partb_method3(double **data, int times){
	struct timespec start;
	struct timespec end;
	clockid_t clock = CLOCK_MONOTONIC;
 	int *mem1 ;
	long back = 64 * MB / sizeof(int);
	*data = malloc(sizeof(tests) / sizeof(int)* sizeof(double));
	for(int k = 0;k< sizeof(tests) / sizeof(int) ; k++){
		(*data)[k] = 0;
	}
	mem1 = malloc (back*sizeof(int));
	for(long k = 0;k<back;k++){
		mem1[k] = 3231;
	}
	
	
	for( int t = 0;t < times ;t++){
		
		for ( int i = 0;i<sizeof(tests)/sizeof(int) - 1;i++){
			
			long n = tests[i]/sizeof(int) - 1;
			clock_gettime(clock, &start);
			for (int j = 0;j<50000;j++){
				mem1[((j*256)&n)] = j+1;
				
			}
			clock_gettime(clock, &end);
			(*data)[i] += timespec_to_nsec(difftimespec(end, start))/(50000);
		}
	}
	long p = 0;
	for(long j = 0;j<back;j++){
		p += mem1[j];
	}
	return p;
}


int dump_output_to_file(double **output, char *filename) {
	FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("could not open output file\n");
		return -1;
	}
	for (int i = 0;i<sizeof(tests)/sizeof(int)-1;i++){
		fprintf(f, "%fns %.2f\n", (*output)[i]/L2BATCH, (float)log2(tests[i]));
		printf ("%fns %.0fKB\n", (*output)[i]/L2BATCH, (float)tests[i]/KB);
	}
	fclose(f);
	printf("The size of the level 1 cache is the largest size the program can test without experience the first significant time increase\n");
	printf("The size of the level 2 cache is the largest size the program can test without experience the second significant time increase\n");
	printf("The size of the level 3 cache is the largest size the program can test without experience the third significant time increase\n");
	printf("The timing data is outputed to stdout, written to data.txt and also graphed in part1b.pdf\n");
	
	return 0;
}


int main(int argc, char *argv[])
{
	// Pin the thread to a single CPU to minimize effects of scheduling
	// Don't use CPU #0 if possible, it tends to be busier with servicing interrupts
	srandom(time(NULL));
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET((random() ?: 1) % get_nprocs(), &set);
	if (sched_setaffinity(getpid(), sizeof(set), &set) != 0) {
		perror("sched_setaffinity");
		return 1;
	}
	long runs = PARTARUNS;
	float data1 [23];
	for (int i = 0;i<23;i++){
		data1[i] = 0;
	}
	parta(data1, runs);
	parta_analysis(data1, runs);
	double *data;
	partb_method3(&data, L2BATCH);
	dump_output_to_file(&data, "data.txt");
	partb_mem_test(PARTBRUNS);
	free(data);
}
