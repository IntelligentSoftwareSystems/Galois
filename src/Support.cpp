/** Support functions -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/mm/Mem.h"

bool Galois::Runtime::inGaloisForEach = false;

void Galois::Runtime::reportPageAlloc(const char* category) {
  for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x)
      Galois::Runtime::reportStat(nullptr, category, Galois::Runtime::MM::numPageAllocForThread(x));
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
   int nodes = Galois::Runtime::MM::numNumaNodes();
    for (int x = 0; x < nodes; ++x)
      Galois::Runtime::reportStat(nullptr, category, Galois::Runtime::MM::numNumaAllocForNode(x));
 
}
