#include <vector>
#include <functional>

#include "galois/runtime/LevelExecutor.h"
#include "galois/worklists/WorkList.h"

#include "bfs.h"
#include "bfsParallel.h"

class LevelExecBFS: public BFS {

public:

  virtual const std::string getVersion () const { return "using Level-by-Level executor"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    ParCounter numIter;


    // update request for root
    Update first (startNode, 0);

    std::vector<Update> wl;
    wl.push_back (first);

    typedef galois::worklists::dChunkedFIFO<OpFunc::CHUNK_SIZE, Update> C;
    typedef galois::worklists::OrderedByIntegerMetric<GetLevel, C>::with_barrier<true>::type WL_ty;

    galois::runtime::for_each_ordered_level (
        galois::runtime::makeStandardRange (wl.begin (), wl.end ()), 
        GetLevel (), 
        std::less<unsigned> (),
        VisitNhood (graph),
        OpFunc (graph, numIter));

    // galois::for_each (first,
        // OpFunc (graph, numIter),
        // galois::loopname ("bfs-level-exec"),
        // galois::wl<WL_ty> ());


    std::cout << "number of iterations: " << numIter.reduce () << std::endl;


    return numIter.reduce ();
  }


};



int main (int argc, char* argv[]) {
  LevelExecBFS b;
  b.run (argc, argv);
  return 0;
}
