#include <vector>
#include <functional>

#include "Galois/Runtime/OrderedSpeculation.h"

#include "bfs.h"

class SSSPspec: public SSSP {
  




public:

  virtual const std::string getVersion () const { return "Speculative SSSP with optimizations"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    ParCounter numIter;

    // update request for root
    Update first (startNode, 0);

    std::vector<Update> wl;
    wl.push_back (first);

    Galois::Runtime::for_each_ordered_spec (
        Galois::Runtime::makeStandardRange(wl.begin (), wl.end ()),
        Comparator (), 
        VisitNhood (graph),
        OpFuncSpec (graph, numIter),
        std::make_tuple (
          Galois::loopname ("sssp_spec")));


    std::cout << "number of iterations: " << numIter.reduce () << std::endl;


    return numIter.reduce ();
  }


};

int main (int argc, char* argv[]) {
  SSSPspec wf;
  wf.run (argc, argv);
  return 0;
}
