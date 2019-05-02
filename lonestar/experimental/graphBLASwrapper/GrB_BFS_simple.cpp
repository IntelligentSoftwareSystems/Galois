#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>

namespace cll = llvm::cl;

// TODO it should be sparse vector.
template <typename T>
using GrB_Vector = galois::LargeArray<T>;

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

template <typename T, typename K = bool, typename I=uint64_t>
void GrB_assign (GrB_Vector<T>& w,
        const GrB_Vector<K>& mask,
        const int accum, // incomplete type.
        const T x,
        const I *ri,
        const I ni,
        const int desc // incompelete type.
        ) {
    galois::do_all(galois::iterate(0ul, w.size()),
            [&] (uint64_t idx) {
                if (mask[idx]) {
                    w[idx] = x;
                }
            }, galois::loopname("VectorAssignment"),
               galois::steal() );
}

template <typename T>
void GrB_Vector_setElement (GrB_Vector<T>& w,
        const T x,
        uint64_t n) {
    galois::do_all(galois::iterate(0ul, n),
            [&] (uint64_t idx) {
                w[idx] = x;
            }, galois::loopname("SetElement"),
               galois::steal() );
}

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
        const GrB_Vector<T> &u,
        GrB_Matrix &A,
        const void *desc) { // assume original, original, complemented v, replace
    GrB_Vector<T> tmpV;
    tmpV.allocateInterleaved(u.size());
    std::copy(u.begin(), u.end(), tmpV.begin());
    galois::do_all(galois::iterate(0ul, w.size()),
            [&] (uint32_t idx) {
                if (!mask[idx]) {
                    for (auto e : A.edges(idx)) {
                        auto jdx = A.getEdgeDst(e);
                        w[idx] = (w[idx] || (tmpV[jdx] && 1));
                    }
                } else {
                    w[idx] = 0;
                }
            },
            galois::loopname("DVxSPM") );
            //galois::steal() ); // make results inconsistent..
}

template <typename T>
void GrB_Vector_new (GrB_Vector<T> *v,
                    int type, uint64_t size) {
    v->allocateInterleaved(size);
}

template <typename T>
void GrB_Dump_Vector (GrB_Vector<T> &v) {
    std::ofstream dmpfp("label.out");
    for (int i = 0; i < v.size(); i++)
        dmpfp << i << "," << v[i] << "\n";
    dmpfp.close();
}

int main(int argc, char** argv) {
    galois::SharedMemSys G;
    llvm::cl::ParseCommandLineOptions(argc, argv);

    numThreads = galois::setActiveThreads(numThreads);

    GrB_Vector<uint64_t> v;
    GrB_Vector<bool> q;

    GrB_Matrix graph;
    // CSC format.
    galois::graphs::readGraph(graph, filename);

    std::cout << " Input graph is : " << filename << "\n";
    std::cout << " The number of active threads is : " << numThreads << "\n";

    galois::StatTimer bfsTimer("BFStimer");
    bfsTimer.start();

    GrB_Vector_new(&v, 0, graph.size());
    GrB_Vector_new(&q, 0, graph.size());

    GrB_Vector_setElement(v, 0ul, graph.size());
    GrB_Vector_setElement(q, false, graph.size());

    // Start Node
    q[startNode] = true;
    uint64_t level = 1;
    for (int64_t level = 1; level <= graph.size(); level ++) {
        // v<q> = level
        GrB_assign<uint64_t>(v, q, -1, level, (uint64_t *) NULL, graph.size(), -1);

        // successor = ||(q)
        bool anyq = false;
        GrB_reduce(&anyq, NULL, NULL, q, NULL);
        //GrB_print_Vector(q);
        if (!anyq) break;

        // q`[!v] = `q or.and A
        GrB_vxm (q, v, NULL, NULL, q, graph, NULL);
    }

    bfsTimer.stop();

    GrB_Dump_Vector(v);

    return 0;
}
