#include <deque>

#include "bfs.h"

class BFSserialFIFO: public BFS {

  typedef std::deque<GNode> WL_ty;
  typedef BFS Super_ty;

public:
  virtual const std::string getVersion () const { return "Serial FIFO"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {


    WL_ty worklist;
    size_t niter = 0;

    graph.getData (startNode, galois::MethodFlag::UNPROTECTED) = 0;
    worklist.push_back (startNode);

    while (!worklist.empty ()) {

      GNode src = worklist.front ();
      worklist.pop_front ();

      Super_ty::bfsOperator<false> (graph, src, worklist);

      ++niter;
    }

    return niter;
  }
};
int main (int argc, char* argv[]) {
  BFSserialFIFO sf;
  sf.run (argc, argv);
  return 0;
}
