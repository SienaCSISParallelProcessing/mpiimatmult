/* 
   Matrix multiplication example

   MPI version, explicit domain decomposition, distributed B

   Jim Teresco, CS 338, Williams College, CS 341, Mount Holyoke College
   Mon Mar 10 21:35:56 EST 2003

   Updated Fall 2021, CSIS 335, Siena College
*/

#include <stdio.h>
#include <sys/param.h>
#include <netdb.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

/* it's a simple program for now, we'll just put everything in main */
int main(int argc, char *argv[]) {

  /* counters */
  int row, col, k;
  double sum, global_sum;

  /* the matrices */
  double **a, **b, **c, **global_b;
  
  /* global matrix size */
  int size;
  
  /* number of rows of A/C, this process has */
  int mysize;
  
  /* for timing */
  double start, stop;
  
  /* MPI and related */
  int numprocs, rank, rc;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostnamelen;
  int pid;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  
  if (rank == 0) {
    if (argc < 2) {
      fprintf(stderr,"Usage: %s size\n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size = atoi(argv[1]);
  }

  /* let's see who we are to the "outside world" - what host and what PID */
  MPI_Get_processor_name(hostname,&hostnamelen);
  pid = getpid();
  
  /* say who we are */
  printf("Process %d of %d has pid %5d on %s\n",rank,numprocs,
	 pid,hostname);
  fflush(stdout);

  /* broadcast argument read from command line */
  rc = MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Bcast failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  
  /* require that the number of procs evenly divides the size */
  if (size%numprocs != 0) {
    if (rank == 0) {
      fprintf(stderr,"Number of procs %d must evenly divide size %d\n",
	      numprocs, size);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  /* compute my size, which we know divides evenly */
  mysize = size/numprocs;
  
  /* initialize and allocate matrices, just fill with junk */
  /* here, we distribute A and C by rows, B is replicated */
  
  start = MPI_Wtime();
  
  /* allocate A, B and C, which have mysize rows, size columns */
  a = (double **)malloc(mysize*sizeof(double *));
  b = (double **)malloc(mysize*sizeof(double *));
  c = (double **)malloc(mysize*sizeof(double *));
  
  for (row=0; row<mysize; row++) {
    a[row] = (double *)malloc(size*sizeof(double));
    b[row] = (double *)malloc(size*sizeof(double));
    c[row] = (double *)malloc(size*sizeof(double));
    for (col=0; col<size; col++) {
      a[row][col] = mysize*rank+row-col;
      b[row][col] = mysize*rank+row+col;
    }
  }

  stop = MPI_Wtime();
  if (rank == 0) {
    printf("Initialization took: %f seconds\n", stop-start);
  }
  
  start = MPI_Wtime();

  /* matrix-matrix multiply */

  /* allocate global B matrix, no need to replicate rows we already have */
  global_b = (double **)malloc(size*sizeof(double *));
  for (row=0; row<size; row++) {
    if (row/mysize == rank) {
      global_b[row] = b[row-mysize*rank];
    } else {
      global_b[row] = (double *)malloc(size*sizeof(double));
    }
  }

  /* communicate to fill global B matrix -- each process needs to
     broadcast its rows -- we can do better, but this makes it clear
     what's happening */
  for (row=0; row<size; row++) {
    rc = MPI_Bcast(global_b[row], size, MPI_DOUBLE, row/mysize, 
		   MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS) {
      fprintf(stderr, "MPI_Bcast failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  for (row=0; row<mysize; row++) {
    for (col=0; col<size; col++) {
      
      /* initialize entry */
      c[row][col] = 0;
      
      /* perform dot product */
      for(k=0; k<size; k++) {
	c[row][col] = c[row][col] + a[row][k]*global_b[k][col];
      }
    }
  }
  
  stop = MPI_Wtime();
  if (rank == 0) {
    printf("Multiplication took: %f seconds\n", stop-start);
  }
  
  /* This is here to make sure the optimizing compiler doesn't
     get any big ideas about "optimizing" code away completely */
  /* first accumulate local sums */
  sum=0;
  for (row=0; row<mysize; row++) {
    for (col=0; col<size; col++) {
      sum += c[row][col];
    }
  }

  printf("Local sum of elements of c on %d=%f\n", rank, sum);

  /* reduction to get the global sum */
  rc = MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, 
		  MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Reduce failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  
  if (rank == 0) {
    printf("Global sum of elements of c=%f\n", global_sum);
  }
  
  /* free memory */
  for (row=0; row<mysize; row++) {
    free(a[row]);
    free(b[row]);
    free(c[row]);
  }
  for (row=0; row<size; row++) {
    if (row/mysize != rank)
      free(global_b[row]);
  }
  free(a);
  free(b);
  free(c);
  free(global_b);

  rc = MPI_Finalize();
  return rc;
}
