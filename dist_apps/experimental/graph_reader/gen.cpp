/** GraphReader -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Reads a graph and exits.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

constexpr static const char* const regionname = "GraphReader";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;

static cll::opt<bool> withEdgeData("dataOn", 
                                   cll::desc("Read or don't read edge data"),
                                   cll::init(false));

static cll::opt<bool> iterateIn("iterateIn", 
                                cll::desc("Read as if iterating over in edges"),
                                cll::init(false));


/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

struct NodeData {
  uint32_t dummy;
};

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "GraphReader";
constexpr static const char* const desc = "Reads a Galois graph.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  galois::StatTimer StatTimer_total("TIMER_TOTAL", regionname); 

  StatTimer_total.start();

  if (withEdgeData) {
    distGraphInitialization<NodeData, unsigned>();
  } else {
    distGraphInitialization<NodeData, void>();
  }

  StatTimer_total.stop();

  return 0;
}
