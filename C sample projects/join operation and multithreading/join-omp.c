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

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "hash.h"
#include "join.h"
#include "options.h"

int find_bounds_and_join(const student_record *students, int students_count, const ta_record *tas, int tas_count, int lower_sid, int higher_sid, join_func_t *join_f){
	int student_lower = -1;
	int student_heigher = -1;
	int tas_lower = -1;
	int tas_heigher = -1;
	int i = 0;
	while(students[i].sid < higher_sid && i < students_count){
		if(student_lower == -1 && students[i].sid >= lower_sid){
			student_lower = i;
		}
		i++;
	}
	student_heigher = i;
	i = 0;
	while(tas[i].sid < higher_sid && i < tas_count){
		if(tas_lower == -1 && tas[i].sid >= lower_sid){
			tas_lower = i;
		}
		i++;
	}
	tas_heigher = i;

	if(student_heigher < student_lower || student_lower == -1){
		student_lower = student_heigher;
	}
	if(tas_heigher < tas_lower || tas_lower == -1){
		tas_lower = tas_heigher;
	}
	int count = join_f(students + student_lower, student_heigher - student_lower, tas + tas_lower, tas_heigher - tas_lower);
	return count ;


	
}

int main(int argc, char *argv[])
{
	const char *path = parse_args(argc, argv);
	if (path == NULL) return 1;

	if (!opt_replicate && !opt_symmetric) {
		fprintf(stderr, "Invalid arguments: parallel algorithm (\'-r\' or \'-s\') is not specified\n");
		print_usage(argv);
		return 1;
	}

	if (opt_nthreads <= 0) {
		fprintf(stderr, "Invalid arguments: number of threads (\'-t\') not specified\n");
		print_usage(argv);
		return 1;
	}
	omp_set_num_threads(opt_nthreads);

	int students_count, tas_count;
	student_record *students;
	ta_record *tas;
	if (load_data(path, &students, &students_count, &tas, &tas_count) != 0) return 1;

	int result = 1;
	join_func_t *join_f = opt_nested ? join_nested : (opt_merge ? join_merge : join_hash);

	double t_start = omp_get_wtime();
	int count = 0;
	if (opt_replicate){
		#pragma omp parallel for reduction(+:count) 
		for(int i = 0;i<opt_nthreads;i++){
			int lbound = i * (tas_count/opt_nthreads);
			int ubound = (i+1) * (tas_count/opt_nthreads);
			if (ubound > tas_count || i == opt_nthreads-1){
				ubound = tas_count;
			}	
			count += join_f(students, students_count, tas + lbound, ubound - lbound);
		}
	}
	else if (opt_symmetric){
		int maxsid = students[students_count-1].sid;
		#pragma omp parallel for reduction(+:count) 
		for(int i = 0; i<opt_nthreads;i++){
			int sidlower = i * maxsid / opt_nthreads;
			int sidhigher = (i+1) * maxsid / opt_nthreads;
			if (i == opt_nthreads-1){
				sidhigher = maxsid;
			}
			count += find_bounds_and_join(students, students_count, tas, tas_count, sidlower, sidhigher, join_f);
			
		}

	}
	#pragma omp barrier
	double t_end = omp_get_wtime();

	if (count < 0) goto end;
	printf("%d\n", count);
	printf("%f\n", (t_end - t_start) * 1000.0);
	result = 0;

end:
	free(students);
	free(tas);
	return result;
}
