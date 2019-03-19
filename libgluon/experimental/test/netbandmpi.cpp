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

#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <unistd.h>

#include <mpi.h>

#include "galois/Timer.h"

volatile int cont = 0;

int main(int argc, char** argv) {
  int trials = 1000000;
  if (argc > 1)
    trials = atoi(argv[1]);

  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

  int numTasks, taskRank;
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);

  if (numTasks != 2) {
    std::cerr << "Just run with 2 hosts\n";

    return 1;
  }

  //  while (!cont) {}

  for (int s = 10; s < trials; s *= 1.1) {
    std::vector<char> vec(s);
    galois::Timer T1, T2, T3;
    MPI_Barrier(MPI_COMM_WORLD);
    T3.start();
    T1.start();
    T1.stop();
    MPI_Barrier(MPI_COMM_WORLD);
    T2.start();
    MPI_Status status;
    if (taskRank == 0) {
      MPI_Request cur;
      MPI_Isend(vec.data(), vec.size(), MPI_BYTE, 1, 0, MPI_COMM_WORLD, &cur);
      int flag;
      do {
        MPI_Test(&cur, &flag, &status);
      } while (!flag);
    } else {
      int nbytes, flag;
      do {
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
      } while (!flag);
      MPI_Get_count(&status, MPI_CHAR, &nbytes);
      MPI_Recv(vec.data(), nbytes, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG,
               MPI_COMM_WORLD, &status);
    }
    T2.stop();
    MPI_Barrier(MPI_COMM_WORLD);
    T3.stop();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cerr << "H" << taskRank << " size " << s << " T1 " << T1.get()
              << " T2 " << T2.get() << " T3 " << T3.get() << " B "
              << (T3.get() ? (s / (T3.get() - T1.get())) : 0.0) << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return 0;
}
