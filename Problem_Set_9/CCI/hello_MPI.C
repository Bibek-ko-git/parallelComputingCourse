#include <iostream>
#include <stdio.h>
#include <mpi.h> 

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char procName[MPI_MAX_PROCESSOR_NAME];
  int resultlen;
  MPI_Get_processor_name( procName, &resultlen);
  printf("Hello World! procName=%s, myRank=%2d, numProcs=%2d\n",procName, rank, size);
  fflush(stdout);

  MPI_Finalize();

  return 0;
}
