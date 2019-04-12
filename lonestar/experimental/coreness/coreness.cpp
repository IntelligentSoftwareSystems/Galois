#include <iostream>
#include <set>
#include <fstream>
#include "galois/AtomicHelpers.h"
#include "galois/DynamicBitset.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include "llvm/Support/CommandLine.h"
#include "galois/graphs/TypeTraits.h"

namespace cll = llvm::cl;

enum Algo {
    naive = 0,
    peeling,
    hindex
};

static cll::opt<std::string>
    inputFName(cll::Positional, 
            cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int>
    numThreads("t", cll::desc("Number of threads"), cll::init(56));
static cll::opt<Algo>
    algo("algo", cll::desc("Choose an algorithm:"),
            cll::values(clEnumVal(naive, "naive"),
                       clEnumVal(peeling, "peeling"),
                       clEnumVal(hindex, "hindex"), clEnumValEnd),
            cll::init(hindex));
static cll::opt<unsigned int>
    outputP("o", cll::desc("Do you want to print outputs?"), cll::init(0));

typedef uint64_t DegreeTy;
typedef std::atomic<DegreeTy> aDegreeTy;
//! Per-Thread-Storage Declaration
typedef galois::gstl::Vector<DegreeTy> VecTy;
typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
//! Per-Thread-Storage Declaration


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
using DynamicBitSet = galois::DynamicBitSet;


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
            galois::loopname("Initialization") );
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
            galois::loopname("Initialization") );
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
            galois::loopname("Initialization") );
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
            galois::steal() );

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
            galois::steal() );

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

bool H(Graph &graph, DynamicBitSet &isDead, GNode cNode,
            ThreadLocalData &nodesThreadLocal) {
    auto& adjHs = *nodesThreadLocal.getLocal();
    adjHs.clear();
    adjHs.resize(std::distance(graph.edge_begin(cNode),
                               graph.edge_end(cNode))+1, 0);

    unsigned int idx;
    for (auto e : graph.edges(cNode)) {
        GNode neigh = graph.getEdgeDst(e);
        idx = std::min<DegreeTy>(kcoreMap[cNode], kcoreMap[neigh]);
        adjHs[idx]++;
    }

    DegreeTy sum = 0, curH = kcoreMap[cNode];
    while (curH >= 1) {
        sum += adjHs[curH];
        if (sum >= curH) break;
        curH--;
    }

    if (kcoreMap[cNode] == curH) {
        return false;
    } else if (kcoreMap[cNode] > curH) {
        kcoreMap[cNode] = curH;
    }

    for (auto e : graph.edges(cNode)) {
        GNode neigh = graph.getEdgeDst(e);
        isDead.reset(neigh);
    }

    return true;
}

/**
 * hIndexCoreness()
 * @graph: Target graph.
 *
 * Find coreness for each node by exploiting h-index method.
 *
 * Return: Maximal coreness value.
 */
DegreeTy hIndexCoreness(Graph &graph, ThreadLocalData &nodesThreadLocal) {
    aDegreeTy maxDegree(MIN_DEGREE);
    DynamicBitSet isDead;
    isDead.resize(graph.size());

    /* Initially, put all nodes onto the current bag */
    galois::do_all(galois::iterate(graph), [&](GNode node) {
            kcoreMap[node] = std::distance(graph.edge_begin(node),
                                           graph.edge_end(node)); },
            galois::loopname("HIndexCorenessInitialization"),
            galois::no_conflicts(),
            galois::steal());

    bool isChange;
    do {
        isChange = false;
        galois::do_all(galois::iterate(graph.begin(), graph.end()),
                [&](GNode cNode) {
                    if (!isDead.test(cNode)) {
                        isDead.set(cNode);
                        if (H(graph, isDead, cNode, nodesThreadLocal))
                            isChange = true;
                    }
                },
                galois::loopname("HIndexCoreness"),
                galois::steal(),
                galois::chunk_size<CHUNK_SIZE>() );

        if (!isChange) break;
    } while(true);

    return maxDegree;
}

int main(int argc, char** argv) {
    galois::SharedMemSys G;
    ThreadLocalData nodesThreadLocal;

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
    /*
    galois::preAlloc((numThreads + (graph.size() * sizeof(unsigned) * 2) /
                galois::runtime::pagePoolSize()) * 8);
                */

    galois::StatTimer Tmain("RunningTime");
    Tmain.start();

    switch (algo) {
        case naive:
            coreness = naiveCoreness(graph);
        break;
        case peeling:
            coreness = peelingCoreness(graph, initBag);
        break;
        case hindex:
            coreness = hIndexCoreness(graph, nodesThreadLocal);
        break;
    }

    Tmain.stop();

    std::cout << "Graph [" << inputFName << "] has coreness of "
        << coreness << std::endl;
    if (outputP) {
        std::ofstream fp;
        fp.open("kcore_output", std::ofstream::out);
        for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ii++) {
            fp << *ii << "," << kcoreMap[*ii] << std::endl;
//            std::cout << *ii << "," << kcoreMap[*ii] << std::endl;
        }
        fp.close();
    }

    delete kcoreMap;
    return 0;
}
