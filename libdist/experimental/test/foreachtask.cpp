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

#include "galois/ForeachTask.h"
#include "galois/runtime/DistributedStructs.h"
#include <vector>

using namespace std;
using namespace galois::runtime;

struct R : public galois::runtime::Lockable {
  int i;

  R() { i = 0; }

  void add(int v) {
    i += v;
    return;
  }
};

int main(int argc, char* argv[]) {
  /*
    int  rc;

    rc = MPI_Init(&argc,&argv);
    if (rc != MPI_SUCCESS) {
      printf ("Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
    }
   */

  galois::setActiveThreads(4);

  // check the task id and decide if the following should be executed
  galois::for_each_begin();

  vector<int> myvec;
  typedef vector<int>::iterator IterTy;
  f1 f;
  for (int i = 1; i <= 40; i++)
    myvec.push_back(i);
  galois::for_each_task<IterTy, f1>(myvec.begin(), myvec.end(), f);
  printf("final output: %d\n", f.r->i);

  master_terminate();

  return 0;
}
