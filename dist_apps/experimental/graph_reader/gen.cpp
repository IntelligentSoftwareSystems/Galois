#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
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

  // iterate over out edges
  if (!iterateIn) {
    if (withEdgeData) {
      distGraphInitialization<NodeData, unsigned>();
    } else {
      distGraphInitialization<NodeData, void>();
    }
  } else {
  // iterate over in edges
    if (withEdgeData) {
      distGraphInitialization<NodeData, unsigned, false>();
    } else {
      distGraphInitialization<NodeData, void, false>();
    }
  }

  StatTimer_total.stop();

  return 0;
}
