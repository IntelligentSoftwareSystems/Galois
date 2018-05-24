#include <vector>
#include <functional>

#include "galois/runtime/KDGspecLocalMin.h"

#include "bfs.h"
#include "bfsParallel.h"

class SpecOptBFS: public BFS {
  
public:

  virtual const std::string getVersion () const { return "Speculative with optimizations"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    ParCounter numIter;

    // update request for root
    Update first (startNode, 0);

    std::vector<Update> wl;
    wl.push_back (first);

    galois::runtime::for_each_ordered_kdg_spec_local_min (
        galois::runtime::makeStandardRange(wl.begin (), wl.end ()),
        Comparator (), 
        VisitNhood (graph),
        OpFuncLocalMin (graph, numIter),
        std::make_tuple (
          galois::loopname ("bfs_spec_local_min")));


    std::cout << "number of iterations: " << numIter.reduce () << std::endl;


    return numIter.reduce ();
  }


};

int main (int argc, char* argv[]) {
  SpecOptBFS wf;
  wf.run (argc, argv);
  return 0;
}
