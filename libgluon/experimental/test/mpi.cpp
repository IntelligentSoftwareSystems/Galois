/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

/************************************************************
This is a simple isend/ireceive program in MPI
************************************************************/

int main(int argc, char** argv) {
  int myid, numprocs;
  int tag, source, destination, count;
  int buffer;
  MPI_Status status;
  MPI_Request request;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  tag         = 1234;
  source      = 0;
  destination = 1;
  count       = 1;
  request     = MPI_REQUEST_NULL;
  if (myid == source) {
    buffer = 5678;
    MPI_Isend(&buffer, count, MPI_INT, destination, tag, MPI_COMM_WORLD,
              &request);
  }
  if (myid == destination) {
    MPI_Irecv(&buffer, count, MPI_INT, source, tag, MPI_COMM_WORLD, &request);
  }
  MPI_Wait(&request, &status);
  if (myid == source) {
    printf("processor %d  sent %d\n", myid, buffer);
  }
  if (myid == destination) {
    printf("processor %d  got %d\n", myid, buffer);
  }
  MPI_Finalize();
  return 0;
}
