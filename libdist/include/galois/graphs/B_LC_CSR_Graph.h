#ifndef __B_LC_CSR_GRAPH__
#define __B_LC_CSR_GRAPH__

#include "galois/graphs/LC_CSR_Graph.h"

namespace galois {
namespace graphs {

template<typename NodeTy, typename EdgeTy,
         bool HasNoLockable=false,
         bool UseNumaAlloc=false,
         bool HasOutOfLineLockable=false,
         typename FileEdgeTy=EdgeTy>
class B_LC_CSR_Graph :
     public LC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc, 
                  HasOutOfLineLockable, FileEdgeTy> {
  // typedef to make it easier to read
  using BaseCSRGraph = LC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc, 
                                 HasOutOfLineLockable, FileEdgeTy>;
protected:
  // retypedefs of base class
  typedef LargeArray<EdgeTy> EdgeData;
  typedef LargeArray<uint32_t> EdgeDst;
  typedef LargeArray<uint64_t> EdgeIndData;

  EdgeIndData inEdgeIndData;
  EdgeDst inEdgeDst;
  EdgeData inEdgeData; 
public:
  B_LC_CSR_Graph() = default;
  B_LC_CSR_Graph(B_LC_CSR_Graph&& rhs) = default;
  B_LC_CSR_Graph& operator=(B_LC_CSR_Graph&&) = default;

  /**
   * Call only after the LC_CSR_Graph is fully constructed.
   * Creates the in edge data.
   */
  void constructIncomingEdges() {
    // TODO
  }
};

} // end graphs namespace
} // end galois namespace
#endif
