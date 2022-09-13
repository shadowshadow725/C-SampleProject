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
#include <string>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include "kernels.h"
#include "pgm.h"

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out) {
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu / time_gpu_computation);
    printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                    time_gpu_transfer_out));
}
static inline double timespec_to_msec(struct timespec t)
{
	return t.tv_sec * 1000.0 + t.tv_nsec / 1000000.0;
}

static inline struct timespec difftimespec(struct timespec t1, struct timespec t0)
{
	assert(t1.tv_nsec < 1000000000);
	assert(t0.tv_nsec < 1000000000);

	return (t1.tv_nsec >= t0.tv_nsec)
		? (struct timespec){ t1.tv_sec - t0.tv_sec    , t1.tv_nsec - t0.tv_nsec             }
		: (struct timespec){ t1.tv_sec - t0.tv_sec - 1, t1.tv_nsec - t0.tv_nsec + 1000000000};
}

int main(int argc, char **argv) {
    int c;
    std::string input_filename, cpu_output_filename, base_gpu_output_filename;
    if (argc < 3) {
        printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
        return 0;
    }

    while ((c = getopt(argc, argv, "i:o:")) != -1) {
        switch (c) {
        case 'i':
        input_filename = std::string(optarg);
        break;
        case 'o':
        cpu_output_filename = std::string(optarg);
        base_gpu_output_filename = std::string(optarg);
        break;
        default:
        return 0;
        }
    }

    pgm_image source_img;
    init_pgm_image(&source_img);

    if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR) {
        printf("Error loading source image.\n");
        return 0;
    }

    /* Do not modify this printf */
    printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
            "Speedup_noTrf Speedup\n");

    double CPU_time = 0.0;
    int8_t lp3_m[] = {
        0, 1, 0, 1, -4, 1, 0, 1, 0,
    };
    {
        struct timespec start;
        struct timespec end;
        clockid_t clock = CLOCK_MONOTONIC;	
        std::string cpu_file = cpu_output_filename;
        pgm_image cpu_output_img;
        copy_pgm_image_size(&source_img, &cpu_output_img);
        // Start time
        clock_gettime(clock, &start);
        run_best_cpu(lp3_m, 3, source_img.matrix, cpu_output_img.matrix, source_img.width, source_img.height);  // From kernels.h
        clock_gettime(clock, &end);
        CPU_time = timespec_to_msec(difftimespec(end, start));
        save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
        destroy_pgm_image(&cpu_output_img);
    }

    {
        std::string gpu_file = "1" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        run_kernel1(lp3_m, 3, source_img.matrix, gpu_output_img.matrix, source_img.width, source_img.height, CPU_time);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        std::string gpu_file = "2" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        run_kernel2(lp3_m, 3, source_img.matrix, gpu_output_img.matrix, source_img.width, source_img.height, CPU_time);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        std::string gpu_file = "3" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        run_kernel3(lp3_m, 3, source_img.matrix, gpu_output_img.matrix, source_img.width, source_img.height, CPU_time);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        std::string gpu_file = "4" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        run_kernel4(lp3_m, 3, source_img.matrix, gpu_output_img.matrix, source_img.width, source_img.height, CPU_time);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }
    {
        std::string gpu_file = "5" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        run_kernel5(lp3_m, 3, source_img.matrix, gpu_output_img.matrix, source_img.width, source_img.height, CPU_time);
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        destroy_pgm_image(&gpu_output_img);
    }

}
