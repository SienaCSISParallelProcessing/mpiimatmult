# Makefile for MPI Matrix Multiplication example, explicit domain decomp
#
# Jim Teresco, CS 338, Williams College
# CS 341, Mount Holyoke College
# CS 400/335, Siena College
#
CFILES=matmult_mpi_toosimple.c
OFILES=$(CFILES:.c=.o)
CC=mpicc

matmult_mpi_toosimple:	$(OFILES)
	$(CC) -o matmult_mpi_toosimple $(OFILES)

clean::
	/bin/rm -f matmult_mpi_toosimple $(OFILES)
