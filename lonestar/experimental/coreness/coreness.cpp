#include <iostream>
#include "galois/AtomicHelpers.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include "llvm/Support/CommandLine.h"
#include "galois/graphs/TypeTraits.h"

namespace cll = llvm::cl;

enum Algo {
    naive = 0,
    peeling
};

static cll::opt<std::string>
    inputFName(cll::Positional, 
            cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int>
    numThreads("t", cll::desc("Number of threads"), cll::init(1));
static cll::opt<Algo>
    algo("algo", cll::desc("Choose an algorithm:"),
            cll::values(clEnumVal(naive, "naive"),
                       clEnumVal(peeling, "peeling")),
            cll::init(naive));

typedef uint64_t DegreeTy;
typedef std::atomic<DegreeTy> aDegreeTy;

DegreeTy* kcoreMap;

constexpr static const DegreeTy MAX_DEGREE =
            std::numeric_limits<DegreeTy>::max();
constexpr static const DegreeTy MIN_DEGREE =
            std::numeric_limits<DegreeTy>::min();
//! Chunksize for for_each worklist: best chunksize will depend on input
constexpr static const unsigned CHUNK_SIZE = 64u;

using Graph =
    galois::graphs::LC_CSR_Graph<aDegreeTy, void>::with_no_lockable<true>::type;
using GNode = Graph::GraphNode;

/**
 * initialize()
 * @graph: Target graph.
 *
 */
void initialize(Graph &graph) {
    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode){
                //NodeData& cNData = graph.getData(cNode);
                aDegreeTy& cNDegree = graph.getData(cNode);
                cNDegree.store(
                    std::distance(graph.edge_begin(cNode),
                                  graph.edge_end(cNode)));
            },
            galois::loopname("Initialization"),
            galois::no_stats() );
}

/**
 * initialize()
 * @graph: Target graph.
 *
 * Not only initialize node data,
 * but also find minimum or maximum degrees.
 */
void initialize(Graph &graph, aDegreeTy& maxDegree, aDegreeTy& minDegree) {
    minDegree = MAX_DEGREE;
    maxDegree = MIN_DEGREE;
    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode) {
               aDegreeTy& cNDegree = graph.getData(cNode);
               cNDegree.store(
                   std::distance(graph.edge_begin(cNode),
                                 graph.edge_end(cNode)));
               galois::atomicMin(minDegree, (DegreeTy) cNDegree);
               galois::atomicMax(maxDegree, (DegreeTy) cNDegree);
            },
            galois::loopname("Initialization"),
            galois::no_stats() );
}

/**
 * initialize()
 * @graph: Target graph.
 *
 * Not only initialize node data,
 * but also find minimum or maximum degrees.
 * In addition, an initial bag is filled.
 */
void initialize(Graph &graph, aDegreeTy& maxDegree,
                  aDegreeTy& minDegree, galois::InsertBag<GNode>& curr) {

    minDegree = MAX_DEGREE;
    maxDegree = MIN_DEGREE;
    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode) {
               aDegreeTy& cNDegree = graph.getData(cNode);
               cNDegree.store(
                   std::distance(graph.edge_begin(cNode),
                                 graph.edge_end(cNode)));
               galois::atomicMin(minDegree, (DegreeTy) cNDegree);
               galois::atomicMax(maxDegree, (DegreeTy) cNDegree);
               curr.push(cNode);
            },
            galois::loopname("Initialization"),
            galois::no_stats() );
}

/**
 * parFindKCore()
 * @graph: Target graph.
 * @k: target degree for k-cores
 *
 * Find whether the graph has k-cores or not in parallel.
 * NOTE: naive version.
 *
 * Return: If it succeeds to find the k-cores,
 *         returns true. Otherwise, return false.
 */
uint8_t parFindKCore(Graph &graph, DegreeTy k) {
    uint8_t isFind = false;
    galois::InsertBag<GNode> cur;

    // Update alive nodes to min degree.
    // Initialize candidnate nodes which are possible to be removed.
    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode) {
                aDegreeTy& cNDegree = graph.getData(cNode);
                if (cNDegree >= k) {
                    kcoreMap[cNode] = k;
                    isFind = true;
                    if (cNDegree == k)
                        cur.emplace(cNode);
                }
            },
            galois::steal(),
            galois::no_stats() );

    // Perform trimming. 
    galois::for_each(galois::iterate(cur.begin(), cur.end()),
            [&](GNode cNode, auto& ctx) {
                /* If the current node has degress that
                   lower than k,
                   then ignores and removes from
                   neighbor sets. */
                for (auto e : graph.edges(cNode)) {
                    GNode neigh = graph.getEdgeDst(e);
                    aDegreeTy& nghDegree = graph.getData(neigh);
                    auto oldVal = galois::atomicSubtract(
                                            nghDegree, 1ul);
                    if (oldVal == k+1) {
                        ctx.push(neigh);
                    }
                }
            },
            galois::no_conflicts(),
            galois::chunk_size<CHUNK_SIZE>() );

    return isFind;
}

/**
 * parFindKCore()
 * @graph: Target graph.
 * @k: target degree for k-cores
 *
 * Find whether the graph has k-cores or not in parallel.
 * NOTE: optimized version of peeling algorithm. It does not use flag.
 *
 * Return: If it succeeds to find the k-cores,
 *         returns true. Otherwise, return false.
 */
int8_t parFindKCore(Graph &graph, DegreeTy k, 
                   aDegreeTy& nextK,
                   galois::InsertBag<GNode>& prev) {
    galois::InsertBag<GNode> next, curr;

    // Update alive nodes to min degree.
    // Initialize candidnate nodes which are possible to be removed.
    // Do not iterate all nodes, but candidate nodes.
    galois::do_all(galois::iterate(prev.begin(), prev.end()),
            [&](GNode cNode) {
                aDegreeTy& cNDegree = graph.getData(cNode);
                if (cNDegree >= k) {
                    if (cNDegree == k)
                        curr.emplace(cNode);
                    else if (cNDegree > k) {
                        next.emplace(cNode);
                        galois::atomicMin(nextK, (DegreeTy) cNDegree);
                    }
                }
            },
            galois::steal(),
            galois::no_stats() );

    // Perform trimming. 
    galois::for_each(galois::iterate(curr.begin(), curr.end()),
            [&](GNode cNode, auto& ctx) {
                /* If the current node has degress that
                   lower than k,
                   then ignores and removes from
                   neighbor sets. */
                kcoreMap[cNode] = k;
                for (auto e : graph.edges(cNode)) {
                    GNode neigh = graph.getEdgeDst(e);
                    aDegreeTy& nghDegree = graph.getData(neigh);
                    auto oldVal = galois::atomicSubtract(
                                        nghDegree, 1ul);
                    if (oldVal == k+1) {
                        ctx.push(neigh);
                    } else if (oldVal > k+1) {
                        galois::atomicMin(nextK, (DegreeTy) (oldVal-1));
                    }
                }
            },
            galois::no_conflicts(),
            galois::chunk_size<CHUNK_SIZE>() );

    // Swap next to prev bag for the next phase.
    prev.clear();
    std::swap(prev, next);

    return (prev.begin() != prev.end());
}

/**
 * naiveCoreness()
 * @graph: Target graph.
 *
 * Iterate from 0 to a maximal possible k value for kcores 
 * (= the number of edges),
 * and check whether there exist any kcores.
 */
DegreeTy naiveCoreness(Graph &graph) {
    DegreeTy cand_k = 0;

    /* graph always need to be initialized */
    initialize(graph);

    for (; cand_k != (DegreeTy) graph.sizeEdges(); ++cand_k) {
        if (!parFindKCore(graph, cand_k)) {
            cand_k--;
            break;
        }
    }

    /* return maximal coreness number among nodes */
    return cand_k;
}

/**
 * peelingCoreness()
 * @graph: Target graph.
 *
 * Iterate from 0 to a maximal possible k value for kcores 
 * (= the number of edges),
 * and check whether there exist any kcores.
 * NOTE: optimized peeling algorithm.
 */
DegreeTy peelingCoreness(Graph &graph, galois::InsertBag<GNode>& curr) {
    DegreeTy cand_k;
    aDegreeTy maxDegree;
    aDegreeTy minDegree;

    initialize(graph, maxDegree, minDegree, curr);
    cand_k = 0;
    do {
        cand_k = minDegree;
        minDegree = MAX_DEGREE;
        if(!parFindKCore(graph, cand_k, minDegree, curr)) break;

    } while (minDegree <= maxDegree);

    /* return maximal coreness number among nodes */
    return cand_k;
}

int main(int argc, char** argv) {
    galois::SharedMemSys G;
    Graph graph;
    galois::InsertBag<GNode> initBag;
    int coreness = 0;

    cll::ParseCommandLineOptions(argc, argv);

    std::cout << "Reading from file: " << inputFName << std::endl;
    galois::graphs::readGraph(graph, inputFName);
    std::cout << "Read " << graph.size() << " nodes, "
        << graph.sizeEdges() << " edges" << std::endl;
    /* maintain kcore numbers for each node */
    kcoreMap = new DegreeTy[graph.size()];

    numThreads = galois::setActiveThreads(numThreads);

    galois::do_all(galois::iterate(0ul, (unsigned long) graph.size()),
                        [&](unsigned long i){ kcoreMap[i] = 0; });

    // preallocate
    //galois::preAlloc((numThreads + (graph.size() * sizeof(unsigned) * 2) /
    //            galois::runtime::pagePoolSize()) * 8); */

    galois::StatTimer Tmain;
    Tmain.start();

    switch (algo) {
        case naive:
            coreness = naiveCoreness(graph);
        break;
        case peeling:
            coreness = peelingCoreness(graph, initBag);
        break;
    }

    Tmain.stop();
    std::cout << "Graph [" << inputFName << "] has coreness of "
        << coreness << std::endl;

    for (size_t node = 0; node != graph.size(); node++) {
        std::cout << node << "," << kcoreMap[node] << std::endl;
    }

    delete kcoreMap;
    return 0;
}
