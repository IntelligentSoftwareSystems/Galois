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

#define GASNET_PAR
#include <gasnet.h>

#include <iostream>

volatile int num;

void remote_short(gasnet_token_t token, gasnet_handlerarg_t arg0) {
  std::cout << "message from " << arg0
            << "\n"; //" at " << gasnet_mynode() << "\n";
  __sync_fetch_and_sub(&num, 1);
}

int main(int argc, char** argv) {
  int res = gasnet_init(&argc, &argv);
  if (res != GASNET_OK)
    std::cerr << "ERROR (" << res << ") when calling gasnet_init\n";

  std::cout << getpid() << " " << gasnet_mynode() << " " << gasnet_nodes()
            << " " << gasnet_getMaxLocalSegmentSize() << " "
            << gasnet_getMaxGlobalSegmentSize() << "\n";

  gasnet_handlerentry_t funcs[1] = {{0, (void (*)())remote_short}};

  res =
      gasnet_attach(funcs, 1, gasnet_getMaxLocalSegmentSize(), GASNET_PAGESIZE);
  if (res != GASNET_OK)
    std::cerr << "ERROR (" << res << ") when calling gasnet_attach\n";
  std::cout << "func at " << (int)funcs[0].index << " on " << gasnet_mynode()
            << "\n";

  num = gasnet_nodes();

  gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
  res = gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
  if (res != GASNET_OK)
    std::cerr << "ERROR (" << res << ") when calling gasnet_barrier_wait\n";

  for (int i = 0; i < gasnet_nodes(); ++i) {
    std::cout << "Sending from " << gasnet_mynode() << " to "
              << (gasnet_mynode() + i) % gasnet_nodes() << " using "
              << (int)funcs[0].index << "\n";
    res = gasnet_AMRequestShort1((gasnet_mynode() + i) % gasnet_nodes(),
                                 funcs[0].index, gasnet_mynode());
    if (res != GASNET_OK)
      std::cerr << "ERROR (" << res
                << ") when calling gasnet_AMRequestShort1\n";
  }

  while (num) {
    res = gasnet_AMPoll();
    if (res != GASNET_OK)
      std::cerr << "ERROR (" << res << ") when calling gasnet_AMPoll\n";
  }

  gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
  res = gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
  if (res != GASNET_OK)
    std::cerr << "ERROR (" << res << ") when calling gasnet_barrier_wait\n";

  gasnet_exit(0);
  return 0;
}
