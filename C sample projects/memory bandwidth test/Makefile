# ------------
# This code is provided solely for the personal and private use of
# students taking the CSC367H5 course at the University of Toronto.
# Copying for purposes other than this use is expressly prohibited.
# All forms of distribution of this code, whether as given or with
# any changes, are expressly prohibited.
#
# Authors: Bogdan Simion, Alexey Khrabrov
#
# All of the files in this directory and all subdirectories are:
# Copyright (c) 2019 Bogdan Simion
# -------------

CC = gcc
CFLAGS += -std=gnu11 -Wall -Werror -g3 -O3 -DNDEBUG -lm

all: part1

part1: part1.o
	$(CC) $^ -o $@ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
#TODO: run your script that runs the experiments, collects all necessary data, and produces the graphs
run: all
	./part1.sh
	

clean:
	rm -f *.o part1
	rm data.txt
	rm part1b.pdf
