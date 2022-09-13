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

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include "hash.h"
#include "join.h"


int join_nested(const student_record *students, int students_count, const ta_record *tas, int tas_count)
{
	assert(students != NULL);
	assert(tas != NULL);
	int count = 0;
		
	for(int i = 0;i<tas_count;i++){
		for(int j = 0;j<students_count;j++){
			if (tas[i].sid == students[j].sid && students[j].gpa > 3){
				count++;
				
			}
		}
	}
	
	return count;
}

// Assumes that records in both tables are already sorted by sid
int join_merge(const student_record *students, int students_count, const ta_record *tas, int tas_count)
{
	assert(students != NULL);
	assert(tas != NULL);
	int t = 0, s = 0, count = 0;
	while(t < tas_count && s < students_count){
		if(tas[t].sid < students[s].sid){
			t++;
		}
		else if(tas[t].sid > students[s].sid){
			s++;
		}
		else{
			if (students[s].gpa > 3){
				count++;
				// printf("%d %d\n", tas[t].sid, tas[t].cid);
			}
			t++;
			while(t < tas_count && tas[t].sid == students[s].sid){
				if (students[s].gpa > 3){
					count++;
					// printf("%d %d\n", tas[t].sid, tas[t].cid);
				}
				t++;
			}
			while(t < tas_count && s < students_count && tas[t].sid == students[s].sid){
				if (students[s].gpa > 3){
					count++;
					// printf("%d %d\n", tas[t].sid, tas[t].cid);
				}
				s++;
			}
		}
	}

	return count;
}

int join_hash(const student_record *students, int students_count, const ta_record *tas, int tas_count)
{
	assert(students != NULL);
	assert(tas != NULL);
	int count = 0;
	hash_table_t *tb = hash_create(students_count);
	for(int i = 0;i<students_count;i++){
		hash_put(tb, students[i].sid, students[i].gpa);
	}
	for(int i = 0;i<tas_count;i++){
		if(hash_get(tb, tas[i].sid) > 3){
			count++;
		}
	}
	hash_destroy(tb);
	return count;
}
