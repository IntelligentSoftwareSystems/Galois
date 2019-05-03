/*
(*
(* This code does not support accumulate/semiring/types yet..
(* It is designed only for BFS_simple; all the types and operations are limited.
(*
*/
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>

namespace cll = llvm::cl;

template <typename T>
using GrB_Vector = galois::LargeArray<T>;
using GrB_Index  = uint64_t;
using GrB_Matrix =
    galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_no_lockable<true>::type;
using GNode = GrB_Matrix::GraphNode;

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<int>
    numThreads("t", cll::desc("Number of threads (default value 1)"),
            cll::init(1));

static cll::opt<uint32_t>
    startNode("startNode", cll::desc("Start node (default value 0)"),
            cll::init(0));
template <typename T>
void GrB_print_Vector (GrB_Vector<T> &v) {
    std::cout << "***********" << std::endl;
    for (size_t i = 0; i < v.size(); i++)
        std::cout << i << ":" << v[i] << std::endl;
}

// assign a scalar to subvector.
// in this version, it only supports w<mask>(I) = accum (w(I), x).
template <typename T, typename K=bool>
void GrB_assign (GrB_Vector<T>& w, // input/output vector for results
        GrB_Vector<K>* mask, // optional mask for w, unused if NULL
        const int accum, // incomplete type.
        const T x, // scalar to assign to w(I)
        const GrB_Index *I, // row indices
        const GrB_Index ni, // # of row indices
        const int desc // incompelete type.
        ) {
    galois::do_all(galois::iterate((GrB_Index) 0, ni),
            [&] (GrB_Index idx) {
            if ((*mask)[idx]) w[idx] = x;
            },
            galois::loopname("VectorAssignment"),
            galois::steal() );
}

// assign a scalar to subvector.
// in this version, it only supports w<mask>(I) = accum (w(I), x).
template <typename T>
void GrB_assign (GrB_Vector<T>& w, // input/output vector for results
        void* mask,
        const int accum, // incomplete type.
        const T x, // scalar to assign to w(I)
        const GrB_Index *I, // row indices
        const GrB_Index ni, // # of row indices
        const int desc // incompelete type.
        ) {
    std::cout << "GrB assign with NULL\n";

    galois::do_all(galois::iterate((GrB_Index) 0, ni),
            [&] (GrB_Index idx) { w[idx] = x; },
            galois::loopname("VectorAssignment"),
            galois::steal() );
}

// add a single entry to a vector.
// In our context, just set `x` to w[i].
template <typename T>
void GrB_Vector_setElement (GrB_Vector<T>& w,
        const T x,
        const GrB_Index i) {
    w[i] = x;
}

// reduce a vector to scalar.
// NOTE: there is another version for matrix: not yet implemented.
template <typename T, typename K>
void GrB_reduce (T *c,
        const void *accum,
        const void *monoid, // assume LOR
        const GrB_Vector<K> &u,
        const void *desc) {
    galois::do_all(galois::iterate(0ul, u.size()),
            [&] (uint64_t idx) {
               *c = u[idx] || c;
            }, galois::loopname("Reduce"),
               galois::steal() );
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
void GrB_vxm (GrB_Vector<T> &w,
        GrB_Vector<K> &mask,
        const void *accum, // assume NULL
        const void *semiring, // assume LOR_LAND_BOOL
        GrB_Vector<T> &u,
        GrB_Matrix &A,
        const void *desc) { // assume original, original, complemented v, replace
    // copy operand vector to temporary larray.
    // we use initial `u` array to compute below multiplication.
    // however, if w and u are the same objects, it does not work correctly.
    // also, larray does not support const copy operation.

    GrB_Vector<T> tmpV;
    tmpV.allocateInterleaved(u.size());
    std::copy(u.begin(), u.end(), tmpV.begin());

    uint64_t size = A.size();
    // if replacable, w is initialized.
    galois::do_all(galois::iterate(0ul, size),
            [&] (uint64_t idx) {
                w[idx] = 0;
            }, galois::loopname("InitializeW"), galois::steal() );

    galois::do_all(galois::iterate(0ul, size),
            [&] (uint64_t idx) {
                if (tmpV[idx]) {
                    for (auto e : A.edges(idx)) {
                        auto jdx = A.getEdgeDst(e);
                        if (!mask[jdx]) {
                            w[jdx] = (w[jdx] || 1);
                        }
                    }
                }
            },
            galois::loopname("DVxSPM"),
            galois::steal() );

    tmpV.destroy();
    tmpV.deallocate();
}

// create a vector.
template <typename T>
void GrB_Vector_new (GrB_Vector<T> *v,
                    int type, uint64_t size) {
    v->allocateInterleaved(size);
}

// print out the results.
template <typename T>
void GrB_Dump_Vector (GrB_Vector<T> &v) {
    std::ofstream dmpfp("label.out");
    for (uint64_t i = 0; i < v.size(); i++)
        dmpfp << i << "," << v[i] << "\n";
    dmpfp.close();
}

// return the number of entries in a vector.
template <typename T>
void GrB_Vector_nvals (GrB_Index *nvals,
                       const GrB_Vector<T> &v) {
    *nvals = v.size();
}

// return the number of rows of a matrix.
void GrB_Matrix_nrows (GrB_Index *nrows,
                       const GrB_Matrix &A) {
    *nrows = A.size();
}

int main(int argc, char** argv) {
    galois::SharedMemSys G;
    llvm::cl::ParseCommandLineOptions(argc, argv);

    numThreads = galois::setActiveThreads(numThreads);

    GrB_Vector<uint32_t> v;
    GrB_Vector<bool> q;
    GrB_Matrix A;
    GrB_Index n, nvals;

    galois::graphs::readGraph(A, filename);
    GrB_Matrix_nrows (&n, A);

    std::cout << " Input graph is : " << filename << "\n";
    std::cout << " The number of active threads is : " << numThreads << "\n";

    galois::StatTimer bfsTimer("BFStimer");
    bfsTimer.start();

    // create an empty vector v, and make it dense.
    // v maintains labels for each node.
    GrB_Vector_new(&v, 0, n);
    GrB_assign(v, NULL, -1, 0u, NULL, n, -1);
    GrB_Vector_nvals(&n, v);

    // create a boolean vector q, and set q(s) to true.
    GrB_Vector_new(&q, 0, n);
    GrB_Vector_setElement(q, true, startNode);

    // BFS traversal and label the nodes.
    for (uint32_t level = 1; level <= n; level ++) {
        // v<q> = level
        GrB_assign (v, &q, -1, level, NULL, n, -1);

        // successor = ||(q)
        bool anyq = false;
        GrB_reduce (&anyq, NULL, NULL, q, NULL);
        if (!anyq) break;

        // q`[!v] = `q or.and A
        GrB_vxm (q, v, NULL, NULL, q, A, NULL);
        //GrB_print_Vector(q);
    }

    //GrB_assign (v, &v, -1, v, NULL, n, -1);
    GrB_Vector_nvals (&nvals, v);

    bfsTimer.stop();

    GrB_Dump_Vector(v);

    return 0;
}
