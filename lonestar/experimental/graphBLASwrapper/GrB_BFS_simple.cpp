/*
(*
(* This code does not support accumulate/semiring/types yet..
(* It is designed only for BFS_simple; all the types and operations are limited.
(*
*/
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include "galois/DynamicBitset.h"
#include "GrB_Vector.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>

#define GrB_ALL 0

namespace cll = llvm::cl;

using GrB_Index = uint64_t;
using GrB_Matrix =
    galois::graphs::LC_CSR_Graph<uint32_t,
                                 uint32_t>::with_no_lockable<true>::type;
using GNode = GrB_Matrix::GraphNode;

galois::DynamicBitSet bset;

struct WorkItem {
  GNode node;
  uint32_t dist;
};

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<int>
    numThreads("t", cll::desc("Number of threads (default value 1)"),
               cll::init(1));

static cll::opt<uint32_t> startNode("startNode",
                                    cll::desc("Start node (default value 0)"),
                                    cll::init(0));

// assign to a subvector.
// in this version, it only supports w<mask>(I) = accum (w(I), u).
template <typename T, typename K = bool>
void GrB_assign(
    GrB_Vector<T, GrB_Index>& w,    // input/output vector for results
    GrB_Vector<K, GrB_Index>& mask, // optional mask for w, unused if NULL
    const int accum,                // incomplete type.
    GrB_Vector<T, GrB_Index> u,
    const GrB_Index* I, // row indices
    const GrB_Index ni, // # of row indices
    const int desc      // incompelete type.
) {
  // w.clear();
  /*
  w.resize(ni);
  galois::do_all(galois::iterate(mask),
          [&] (K idx) {
                  w[(T) idx] = u[(T) idx];
          },
          galois::loopname("VVectorAssignment"),
          galois::steal() );
          */
}

// assign a scalar to subvector.
// in this version, it only supports w<mask>(I) = accum (w(I), x).
template <typename T, typename K = bool>
void GrB_assign(GrB_Vector<T, GrB_Index>& w,    // should be sparse
                GrB_Vector<K, GrB_Index>& mask, // should be sparse
                const int accum,                // not supported, (just accum)
                const T x,                      // scalar to assign to w(I)
                const GrB_Index* I,             // not supported
                const GrB_Index ni, // # of row indices, not supported
                const int desc      // not supported (ooco)
) {
  galois::do_all(
      galois::iterate(mask.iterateSPVec()),
      [&](auto& elem) { w.setElement(x, elem.idx); },
      galois::loopname("assign"), galois::steal());
}

// assign a scalar to subvector.
// in this version, it only supports w<mask>(I) = accum (w(I), x).
// assume that I is GrB_ALL
template <typename T>
void GrB_assign(GrB_Vector<T, GrB_Index>& w, // become dense
                void* mask,                  // NULL (GrB_ALL)
                const int accum,             // not supported
                const T x,                   // scalar to assign to w(I)
                const GrB_Index* I,          // not supported
                const GrB_Index ni,          // # of row indices, not supported
                const int desc               // not supported (ooco)
) {
  if (I == GrB_ALL)
    w.trySDConvert();

  // Handling dense vector
  galois::do_all(
      galois::iterate(0lu, w.getSize()),
      [&](GrB_Index idx) { w.setElement(x, idx); }, galois::loopname("assign"),
      galois::steal());
}

// add a single entry to a vector.
// In our context, just set `x` to w[i].
template <typename T, typename K>
void GrB_Vector_setElement(GrB_Vector<T, GrB_Index>& w, const K x,
                           const GrB_Index i) {
  w.setElement(x, i);
}

// reduce a vector to scalar.
// NOTE: there is another version for matrix: not yet implemented.
template <typename T, typename K>
void GrB_reduce(T* c, const void* accum,
                const void* monoid,          // assume LOR
                GrB_Vector<K, GrB_Index>& u, // should sparse
                const void* desc) {
  T& cr = *c;
  if (!u.getSparseVec().empty())
    cr = true;
}

/**
 * GrB_vxm()
 * @w
 * @mask
 * @accum
 * @semiring
 * @u
 * @A
 * @desc
 *
 * Dense vector and sparse matrix multiplication
 *
 */
template <typename T, typename K>
void GrB_vxm(GrB_Vector<T, GrB_Index>& w,    // sparse
             GrB_Vector<K, GrB_Index>& mask, // dense
             const void* accum,              // assume NULL
             const void* semiring,           // assume LOR_LAND_BOOL
             GrB_Vector<T, GrB_Index>& u,    // sparse
             GrB_Matrix& A,                  // CSR
             const void* desc) {
  GrB_Vector<T, GrB_Index> next;
  GrB_Vector_new(&next, 0, w.getSize());
  next.setDupCheckMode();
  galois::do_all(
      galois::iterate(w.iterateSPVec()),
      [&](auto& item) {
        GNode src = item.idx;
        for (auto e : A.edges(src)) {
          auto dst = A.getEdgeDst(e);
          if (!mask.getDenseElement(dst)) {
            next.setElement(true, dst);
          }
        }
      },
      galois::loopname("SVxSPM-checkActive"), galois::steal());
  next.unsetDupCheckMode();

  std::swap(w.getSparseVec(), next.getSparseVec());
  next.clear();
}

// create a vector.
template <typename T>
void GrB_Vector_new(GrB_Vector<T, GrB_Index>* v, int type, uint64_t size) {
  v->Initialize(size);
}

// return the number of entries in a vector.
template <typename T>
void GrB_Vector_nvals(GrB_Index* nvals, GrB_Vector<T, GrB_Index>& v) {
  *nvals = v.getSize();
}

// return the number of rows of a matrix.
void GrB_Matrix_nrows(GrB_Index* nrows, const GrB_Matrix& A) {
  *nrows = A.size();
}

template <typename T>
void GrB_free(T* target) {}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  llvm::cl::ParseCommandLineOptions(argc, argv);

  numThreads = galois::setActiveThreads(numThreads);

  GrB_Vector<uint32_t, GrB_Index> v; // label
  GrB_Vector<GNode, GrB_Index> q;    // active elements
  GrB_Matrix A;
  GrB_Index n, nvals;

  galois::graphs::readGraph(A, filename);

  std::cout << " Input graph is : " << filename << "\n";
  std::cout << " The number of active threads is : " << numThreads << "\n";
  std::cout << " The number of nodes is : " << A.size() << "\n";

  galois::StatTimer mainTimer("MainTimer");
  mainTimer.start();

  GrB_Matrix_nrows(&n, A);
  // create an empty vector v, and make it dense.
  // v maintains labels for each node.
  // arg: vector, type, size
  GrB_Vector_new(&v, 0, n);                 // 4.7.1
  GrB_assign(v, NULL, -1, 0u, NULL, n, -1); // line 113 (doc 7.11.6)
  GrB_Vector_nvals(&n, v);

  // create a boolean vector q, and set q(s) to true.
  GrB_Vector_new(&q, 0, n);
  GrB_Vector_setElement(q, true, startNode);

  galois::StatTimer bfsTimer("BFStimer");
  bfsTimer.start();
  // BFS traversal and label the nodes.
  uint32_t level = 1;
  for (; level <= n; level++) {
    // v<q> = level
    GrB_assign(v, q, -1, level, NULL, n, -1); // line 77

    // successor = ||(q)
    bool anyq = false;
    GrB_reduce(&anyq, NULL, NULL, q, NULL); // line 159
    if (!anyq) {
      break;
    }

    // q`[!v] = `q or.and A
    GrB_vxm(q, v, NULL, NULL, q, A, NULL); // line 192
                                           // GrB_print_Vector(q);
  }
  bfsTimer.stop();
  mainTimer.stop();

  // GrB_assign (v, v, -1, v, NULL, n, -1);
  GrB_Vector_nvals(&nvals, v);

  GrB_free(&q);

  v.dump();

  return 0;
}
