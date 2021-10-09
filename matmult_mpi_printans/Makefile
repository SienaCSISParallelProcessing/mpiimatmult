# Makefile for MPI Matrix Multiplication example, explicit domain decomp,
# print answer at the end
#
# CS 400, Siena College
#
# $Id$
#
CFILES=matmult_mpi_printans.c
OFILES=$(CFILES:.c=.o)
CC=mpicc

matmult_mpi_printans:	$(OFILES)
	$(CC) -o matmult_mpi_printans $(OFILES)

clean::
	/bin/rm -f matmult_mpi_printans $(OFILES)
