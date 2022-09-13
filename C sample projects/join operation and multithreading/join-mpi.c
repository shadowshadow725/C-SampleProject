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

#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "join.h"
#include "options.h"


int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int result = 1;
	student_record *students = NULL;
	ta_record *tas = NULL;
	int parts, id;
	
	MPI_Comm_size(MPI_COMM_WORLD, &parts);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	int *counts = malloc(parts * sizeof(int));
	int displacement [parts];
	const char *path = parse_args(argc, argv);
	if (path == NULL) goto end;

	if (!opt_replicate && !opt_symmetric) {
		fprintf(stderr, "Invalid arguments: parallel algorithm (\'-r\' or \'-s\') is not specified\n");
		print_usage(argv);
		goto end;
	}

	
	// Load this process'es partition of data
	char part_path[PATH_MAX] = "";
	snprintf(part_path, sizeof(part_path), "%s_%d", path, id);
	int students_count, tas_count;
	if (load_data(part_path, &students, &students_count, &tas, &tas_count) != 0) goto end;

	join_func_t *join_f = opt_nested ? join_nested : (opt_merge ? join_merge : join_hash);
	int count = 0;
	
	
	MPI_Barrier(MPI_COMM_WORLD);
	double t_start = MPI_Wtime();
	displacement[0] = 0;
	//TODO: parallel join using MPI
	if (opt_replicate){
		MPI_Barrier(MPI_COMM_WORLD);
		
		MPI_Allgather(&students_count, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
		int total_student_count = 0;
		MPI_Barrier(MPI_COMM_WORLD);
		for (int i = 0;i<parts;i++){
			total_student_count += counts[i];
			if (i > 0){
				displacement[i] = displacement[i-1] +  (counts[i] * sizeof(student_record));
			}
			counts[i] = counts[i] * sizeof(student_record);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		student_record *new_students = (student_record*)malloc(total_student_count * sizeof(*students));
		MPI_Allgatherv(students, (sizeof(student_record) * students_count), MPI_BYTE, new_students, counts, displacement, MPI_BYTE, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		count = join_f(new_students, total_student_count, tas, tas_count);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	}
	else if (opt_symmetric){
		count = join_f(students, students_count, tas, tas_count);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	double t_end = MPI_Wtime();

	if (count < 0) goto end;
	if (id == 0) {
		count = 0;
		for(int i = 0;i<parts;i++){
			count += counts[i];
		}
		printf("%d\n", count);
		printf("%f\n", (t_end - t_start) * 1000.0);
	}
	result = 0;

end:
	if (students != NULL) free(students);
	if (tas != NULL) free(tas);
	MPI_Finalize();
	free(counts);
	return result;
}
