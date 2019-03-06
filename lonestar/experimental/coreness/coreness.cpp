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
    peeling,
    peeling_flg_opt
};

static cll::opt<std::string>
    inputFName(cll::Positional, 
            cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int>
    numThreads("t", cll::desc("Number of threads"), cll::init(1));
static cll::opt<Algo>
    algo("algo", cll::desc("Choose an algorithm:"),
            cll::values(clEnumVal(naive, "naive"),
                       clEnumVal(peeling, "peeling"),
                       clEnumVal(peeling_flg_opt, "peeling_flg_opt")),
            cll::init(naive));

typedef int64_t DegreeTy;

struct NodeData {
    /* current number of valid degress */ 
    std::atomic<DegreeTy> validDegree;
    /* trimmed edges */
    std::atomic<DegreeTy> trim;
    /* possible active elements */
    uint8_t flag;
};
DegreeTy* kcoreMap;

constexpr static const DegreeTy K_INFINITY =
            std::numeric_limits<DegreeTy>::max();

using Graph =
    galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;
using GNode = Graph::GraphNode;

/**
 * initialize()
 * @graph: Target graph.
 *
 */
void initialize(Graph &graph) {
    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode node){
                NodeData& currNode = graph.getData(node);
                currNode.flag   = true;
                currNode.validDegree = 0;
                currNode.trim = 0;

                for (auto edge : graph.edges(node))
                   ++currNode.validDegree;
            } );
}

/**
 * initialize()
 * @graph: Target graph.
 *
 * Not only initialize node data,
 * but also find minimum or maximum degrees.
 */
void initialize(Graph &graph, DegreeTy& maxDegree, 
                    std::atomic<DegreeTy>& minDegree) {
    minDegree = K_INFINITY;
    maxDegree = std::numeric_limits<DegreeTy>::min(); 
    for (auto iter = graph.begin();
            iter != graph.end();
            ++iter) {
        GNode node = *iter;
        NodeData& currNode = graph.getData(node);
        currNode.flag   = true;
        currNode.trim = 0;

        for (auto edge : graph.edges(node))
            ++currNode.validDegree;

        if (minDegree > currNode.validDegree)
            minDegree = (DegreeTy) currNode.validDegree;
        if (maxDegree < currNode.validDegree)
            maxDegree = currNode.validDegree;
    }
}

/**
 * initialize()
 * @graph: Target graph.
 *
 * Not only initialize node data,
 * but also find minimum or maximum degrees.
 * In addition, an initial bag is filled.
 */
void initialize(Graph &graph, DegreeTy& maxDegree, 
                    std::atomic<DegreeTy>& minDegree,
                        galois::InsertBag<GNode>& curr) {
    minDegree = K_INFINITY;
    maxDegree = std::numeric_limits<DegreeTy>::min(); 
    for (auto iter = graph.begin();
            iter != graph.end();
            ++iter) {
        GNode node = *iter;
        NodeData& currNode = graph.getData(node);
        currNode.flag = true;
        currNode.trim = 0;

        for (auto edge : graph.edges(node))
            ++currNode.validDegree;

        if (minDegree > currNode.validDegree)
            minDegree = (DegreeTy) currNode.validDegree;
        if (maxDegree < currNode.validDegree)
            maxDegree = currNode.validDegree;

        curr.push(node);
    }
}

/**
 * copyToKcoreMap()
 * @graph: Target graph.
 *
 * Copy current degrees for each nodes to the kcoreMap.
 * This is because the graph keep changing,
 * while trying to find k+1 cores on the graph.
 */
void copyToKcoreMap(Graph& graph) {
    do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode) {
                NodeData& cNData = graph.getData(cNode);
                kcoreMap[cNode] = cNData.validDegree;
            },
            galois::steal());
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
    using PSchunk = galois::worklists::PerSocketChunkFIFO<16>;
    galois::InsertBag<GNode> initBag;
    auto trimming =
        [&](GNode cNode, auto& ctx) {
            NodeData& cNData = graph.getData(cNode);
            /* If the current node has degress that
               lower than k,
               then ignores and removes from
               neighbor sets. */
                cNData.flag = false;
                for (auto e : graph.edges(cNode)) {
                    GNode neigh = graph.getEdgeDst(e);
                    NodeData& nghData = graph.getData(neigh);
                    auto oldVal = galois::atomicAdd(nghData.validDegree,
                                                        (DegreeTy) -1);
                    if (oldVal == k+1) {
                        ctx.push(neigh);
                    }
                }
        };

    // Update alive nodes to min degree.
    // Initialize candidnate nodes which are possible to be removed.
    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode) {
                NodeData& cNData = graph.getData(cNode);
                if (cNData.flag) {
                    kcoreMap[cNode] = k;
                    if (cNData.validDegree == k)
                        initBag.push(cNode);
                }
            } , galois::steal() );

    // Perform trimming. 
    galois::for_each(galois::iterate(initBag.begin(), initBag.end()),
            trimming,
            galois::wl<PSchunk>() );

    // Find minimum degree.
    galois::do_all(
        galois::iterate(graph.begin(), graph.end()),
        [&](GNode cNode) {
            NodeData& cNData = graph.getData(cNode);
            if (cNData.flag) {
                /* If a node has degrees higher than k,
                   at least it is in the k-core. */
                isFind = true;
           } } );

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
uint8_t parFindKCore(Graph &graph, DegreeTy k, 
                   std::atomic<DegreeTy>& nextK,
                   galois::InsertBag<GNode>& prev) {
    uint8_t isFind = false;
    using PSchunk = galois::worklists::PerSocketChunkFIFO<16>;
    galois::InsertBag<GNode> next, curr;
    auto trimming =
        [&](GNode cNode, auto& ctx) {
            /* If the current node has degress that
               lower than k,
               then ignores and removes from
               neighbor sets. */
            NodeData& cNData = graph.getData(cNode);
            for (auto e : graph.edges(cNode)) {
                GNode neigh = graph.getEdgeDst(e);
                NodeData& nghData = graph.getData(neigh);
                auto oldVal = galois::atomicAdd(nghData.validDegree,
                                                    (DegreeTy) -1);
                if (oldVal == k+1) {
                    ctx.push(neigh);
                }
            }
        };

    // Update alive nodes to min degree.
    // Initialize candidnate nodes which are possible to be removed.
    // Do not iterate all nodes, but candidate nodes.
    galois::do_all(galois::iterate(prev.begin(), prev.end()),
            [&](GNode cNode) {
                NodeData& cNData = graph.getData(cNode);
                DegreeTy nDegree = cNData.validDegree;
                if (nDegree >= k) {
                    kcoreMap[cNode] = k;
                    if (nDegree == k) curr.push(cNode);
                    else next.push(cNode);
                }
            } , galois::steal() );

    // Perform trimming. 
    galois::for_each(galois::iterate(curr.begin(), curr.end()),
            trimming,
            galois::wl<PSchunk>() );

    // Swap next to prev bag for the next phase.
    prev.clear();
    std::swap(prev, next);

    // Find minimum degree.
    galois::do_all(
        galois::iterate(prev.begin(), prev.end()),
        [&](GNode cNode) {
            NodeData& cNData = graph.getData(cNode);
            /* If a node has degrees higher than k,
               at least it is in the k-core. */
            isFind = true;

            if (nextK > cNData.validDegree && cNData.validDegree > k)
                galois::atomicMin(nextK, (DegreeTy) cNData.validDegree);
           } );

    return isFind;
}

/**
 * parFindKCore()
 * @graph: Target graph.
 * @k: target degree for k-cores
 *
 * Find whether the graph has k-cores or not in parallel.
 * NOTE: peeling based.
 *
 * Return: If it succeeds to find the k-cores,
 *         returns true. Otherwise, return false.
 */
uint8_t parFindKCore(Graph &graph, DegreeTy k, std::atomic<DegreeTy>& nextK) {
    uint8_t isFind = false;
    using PSchunk = galois::worklists::PerSocketChunkFIFO<16>;
    galois::InsertBag<GNode> initBag;
    auto trimming =
        [&](GNode cNode, auto& ctx) {
            NodeData& cNData = graph.getData(cNode);
            /* If the current node has degress that
               lower than k,
               then ignores and removes from
               neighbor sets. */
                cNData.flag = false;
                for (auto e : graph.edges(cNode)) {
                    GNode neigh = graph.getEdgeDst(e);
                    NodeData& nghData = graph.getData(neigh);
                    auto oldVal = galois::atomicAdd(nghData.validDegree,
                                                    (DegreeTy) -1);
                    if (oldVal == k+1) {
                        ctx.push(neigh);
                    }
                }
        };

    // Update alive nodes to min degree.
    // Initialize candidnate nodes which are possible to be removed.
    galois::do_all(galois::iterate(graph.begin(), graph.end()),
            [&](GNode cNode) {
                NodeData& cNData = graph.getData(cNode);
                if (cNData.flag) {
                    kcoreMap[cNode] = k;
                    if (cNData.validDegree == k)
                        initBag.push(cNode);
                }
            } , galois::steal() );

    // Perform trimming. 
    galois::for_each(galois::iterate(initBag.begin(), initBag.end()),
            trimming,
            galois::wl<PSchunk>() );

    // Find minimum degree.
    galois::do_all(
        galois::iterate(graph.begin(), graph.end()),
        [&](GNode cNode) {
            NodeData& cNData = graph.getData(cNode);
            if (cNData.flag) {
                /* If a node has degrees higher than k,
                   at least it is in the k-core. */
                isFind = true;
                if (nextK > cNData.validDegree)
                    galois::atomicMin(nextK, (DegreeTy) cNData.validDegree);
           } } );

    return isFind;
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

    for (size_t node = 0; node != graph.size(); node++) {
        std::cout << node << "," << kcoreMap[node] << std::endl;
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
 */
DegreeTy peelingCoreness(Graph &graph) {
    DegreeTy cand_k;
    DegreeTy maxDegree;
    std::atomic<DegreeTy> minDegree;

    initialize(graph, maxDegree, minDegree);
    cand_k = 0;
    do {
        cand_k = minDegree;
        minDegree = K_INFINITY;
        if(!parFindKCore(graph, cand_k, minDegree)) break;

    } while (minDegree <= maxDegree);


    for (size_t node = 0; node != graph.size(); node++) {
        std::cout << node << "," << kcoreMap[node] << std::endl;
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
    DegreeTy maxDegree;
    std::atomic<DegreeTy> minDegree;

    initialize(graph, maxDegree, minDegree, curr);
    cand_k = 0;
    do {
        cand_k = minDegree;
        minDegree = K_INFINITY;
        if(!parFindKCore(graph, cand_k, minDegree, curr)) break;

    } while (minDegree <= maxDegree);


    for (size_t node = 0; node != graph.size(); node++) {
        std::cout << node << "," << kcoreMap[node] << std::endl;
    }

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
            coreness = peelingCoreness(graph);
        break;
        case peeling_flg_opt:
            coreness = peelingCoreness(graph, initBag);
        break;
    }

    Tmain.stop();
    std::cout << "Graph [" << inputFName << "] has coreness of "
        << coreness << std::endl;

    delete kcoreMap;
    return 0;
}
