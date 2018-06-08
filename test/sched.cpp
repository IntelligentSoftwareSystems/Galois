/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#include "galois/Timer.h"
#include "galois/Galois.h"

#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_USE_EXP
#include "galois/worklists/WorkListExperimental.h"
#endif

#include <iostream>

static const char* name = "Scheduler Micro Benchmark";
static const char* desc = "Measures stuff";
static const char* url = 0;

static llvm::cl::opt<int> sval(llvm::cl::Positional, llvm::cl::desc("<start value>"), llvm::cl::init(-1));
static llvm::cl::opt<int> ival(llvm::cl::Positional, llvm::cl::desc("<init num>"), llvm::cl::init(100));

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);

  std::vector<int> v((int)ival, (int)sval);

  std::cout << "Initial: " << (int)ival << " using " << (int)sval << "\n";


  galois::StatTimer T2("T2");
  T2.start();
  using namespace galois::worklists;
  galois::for_each(galois::iterate(v), 
      [&] (int item, auto& lwl) {
        for (int i = 0; i < item; ++i)
          lwl.push(item - 1);
      },
      galois::wl<PerSocketChunkLIFO<64>>());
  T2.stop();
}
