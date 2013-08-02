/** GMetis -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Xin Sui <xinsui@cs.utexas.edu>
 * @author Nikunj Yadav <nikunj@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */


#include "Metis.h"
#include "Galois/Runtime/PerThreadStorage.h"
namespace {

void assertAllMatched(GNode node, GGraph* graph) {
  for (auto jj = graph->edge_begin(node), eejj = graph->edge_end(node);
       jj != eejj; ++jj)
    assert(node == graph->getEdgeDst(jj) || graph->getData(graph->getEdgeDst(jj)).isMatched());
}

void assertNoMatched(GGraph* graph) {
  for (auto nn = graph->begin(), en = graph->end(); nn != en; ++nn)
    assert(!graph->getData(*nn).isMatched());
}

struct HEMmatch {
  GNode operator()(GNode node, GGraph* graph) {
    GNode retval = node; // match self if nothing else
    int maxwgt = std::numeric_limits<int>::min();
    //    nume += std::distance(graph->edge_begin(node), graph->edge_end(node));
    for (auto jj = graph->edge_begin(node, Galois::MethodFlag::NONE), eejj = graph->edge_end(node);
         jj != eejj; ++jj) {
      //      ++checked;
      GNode neighbor = graph->getEdgeDst(jj, Galois::MethodFlag::NONE);
      MetisNode& neighMNode = graph->getData(neighbor, Galois::MethodFlag::NONE);
      int edgeData = graph->getEdgeData(jj, Galois::MethodFlag::NONE);
      if (!neighMNode.isMatched() && neighbor != node && maxwgt < edgeData) {
        maxwgt = edgeData;
        retval = neighbor;
      }
    }
    return retval;
  }
};


struct RMmatch {
  GNode operator()(GNode node, GGraph* graph) {
    for (auto jj = graph->edge_begin(node, Galois::MethodFlag::NONE), eejj = graph->edge_end(node);
         jj != eejj; ++jj) {
      GNode neighbor = graph->getEdgeDst(jj, Galois::MethodFlag::NONE);
      if (!graph->getData(neighbor, Galois::MethodFlag::NONE).isMatched() && neighbor != node)
        return neighbor;
    }
    return node;
    //Don't actually do random, just choose first
  }
};

template<typename MatchingPolicy>
struct TwoHopMatcher {
  MatchingPolicy matcher;
  GNode operator()(GNode node, GGraph* graph) {
    for (auto jj = graph->edge_begin(node, Galois::MethodFlag::NONE), eejj = graph->edge_end(node);
         jj != eejj; ++jj) {
      GNode neighbor = graph->getEdgeDst(jj, Galois::MethodFlag::NONE);
      GNode retval = matcher(neighbor, graph);
      if (retval != node && retval != neighbor)
        return retval;
    }
    return node;
  }
};

/*
 *This operator is responsible for matching.
 1. There are two types of matching. Random and Heavy Edge matching 
 2. Random matching picks any random node above a threshold and matches the nodes. RM.h 
 3. Heavy Edge Matching matches the vertex which is connected by the heaviest edge. HEM.h 
 4. This operator can also create the multinode, i.e. the node which is created on combining two matched nodes.  
 5. You can enable/disable 4th by changing variantMetis::mergeMatching
*/

template<typename MatchingPolicy>
struct parallelMatchAndCreateNodes {
  MatchingPolicy matcher;
  TwoHopMatcher<MatchingPolicy> matcher2;
  GGraph *fineGGraph;
  GGraph *coarseGGraph;
  Galois::InsertBag<GNode>& bag;

  bool with2Hop;
  parallelMatchAndCreateNodes(MetisGraph* Graph, Galois::InsertBag<GNode>& bagNoNeighbor, bool _with2Hop = false)
    : matcher(), matcher2(),
      fineGGraph(Graph->getFinerGraph()->getGraph()), coarseGGraph(Graph->getGraph()), bag(bagNoNeighbor), with2Hop(_with2Hop) {
    assert(fineGGraph != coarseGGraph);
  }

  void operator()(GNode item, Galois::UserContext<GNode> &lwl) {
    if (fineGGraph->getData(item).isMatched())
      return;
    if(fineGGraph->edge_begin(item, Galois::MethodFlag::NONE) == fineGGraph->edge_end(item, Galois::MethodFlag::NONE)){
      bag.push(item);
      return;
    }
    GNode ret;
    do {
      if (with2Hop && fineGGraph->getData(item).isFailedMatch() )
        ret = matcher2(item, fineGGraph);
      else
        ret = matcher(item, fineGGraph);
      //lock ret, since we found it lock-free it may be matched, so try again
    } while (fineGGraph->getData(ret).isMatched());

    //at this point both ret and item (and failed matches) are locked.
    //We do not leave the above loop until we both have the lock on
    //the node and check the matched status of the locked node.  the
    //lock before (final) read ensures that we will see any write to matched

    unsigned numEdges = std::distance(fineGGraph->edge_begin(item, Galois::MethodFlag::NONE), fineGGraph->edge_end(item, Galois::MethodFlag::NONE));
    //assert(numEdges == std::distance(fineGGraph->edge_begin(item), fineGGraph->edge_end(item)));

    GNode N;
    if (ret != item) {
      //match found
      numEdges += std::distance(fineGGraph->edge_begin(ret, Galois::MethodFlag::NONE), fineGGraph->edge_end(ret, Galois::MethodFlag::NONE));
      //Cautious point
      N = coarseGGraph->createNode(numEdges, 
                                   fineGGraph->getData(item).getWeight() +
                                   fineGGraph->getData(ret).getWeight(),
                                   item, ret);
      fineGGraph->getData(item).setMatched();
      fineGGraph->getData(ret).setMatched();
      fineGGraph->getData(item).setParent(N);
      fineGGraph->getData(ret).setParent(N);
    } else {
      //assertAllMatched(item, fineGGraph);
      //Cautious point
      //no match
      if (fineGGraph->getData(item).isFailedMatch() || !with2Hop) {
        N = coarseGGraph->createNode(numEdges, fineGGraph->getData(item).getWeight(), item);
        fineGGraph->getData(item).setMatched();
        fineGGraph->getData(item).setParent(N);
      } else {
        fineGGraph->getData(item).setFailedMatch();
        lwl.push(item);
      }
    }
  }
};

struct weightComp{
  GGraph &graph;
  weightComp(GGraph &_graph): graph(_graph){}
  bool operator() (GNode i, GNode j) { return graph.getData(i).getWeight() < graph.getData(j).getWeight(); }
};

void matchNoNeighborNodes(MetisGraph* Graph, Galois::InsertBag<GNode>& bag){
  GGraph *fineGGraph(Graph->getFinerGraph()->getGraph());
  GGraph *coarseGGraph(Graph->getGraph());
  std::vector<GNode> nodes;

  for( auto ii = bag.begin(), ie = bag.end(); ii != ie; ii++){
    nodes.push_back(*ii);
  }
  std::sort (nodes.begin(), nodes.end(), weightComp(*fineGGraph));
  auto ii = nodes.begin(), ie = nodes.end(); 
  while( ii != ie ){
    ie--;
    GNode ret = *ii;
    GNode item = *ie;
    unsigned numEdges = std::distance(fineGGraph->edge_begin(item, Galois::MethodFlag::NONE), fineGGraph->edge_end(item, Galois::MethodFlag::NONE));
    GNode N;
    if (ret != item) {
      //match found
      numEdges += std::distance(fineGGraph->edge_begin(ret, Galois::MethodFlag::NONE), fineGGraph->edge_end(ret, Galois::MethodFlag::NONE));
      //Cautious point
      N = coarseGGraph->createNode(numEdges, 
                                   fineGGraph->getData(item).getWeight() +
                                   fineGGraph->getData(ret).getWeight(),
                                   item, ret);
      fineGGraph->getData(item).setMatched();
      fineGGraph->getData(ret).setMatched();
      fineGGraph->getData(item).setParent(N);
      fineGGraph->getData(ret).setParent(N);
      ii++;
    } else {
      //Cautious point
      //no match
        N = coarseGGraph->createNode(numEdges, fineGGraph->getData(item).getWeight(), item);
        fineGGraph->getData(item).setMatched();
        fineGGraph->getData(item).setParent(N);
    }

  }
}



/*
 * This operator is responsible for doing a union find of the edges
 * between matched nodes and populate the edges in the coarser graph
 * node.
 */

struct parallelPopulateEdges {
  typedef int tt_does_not_need_push;
  typedef int tt_needs_per_iter_alloc;
    
  GGraph *coarseGGraph;
  GGraph *fineGGraph;
  parallelPopulateEdges(MetisGraph *Graph)
    :coarseGGraph(Graph->getGraph()), fineGGraph(Graph->getFinerGraph()->getGraph()) {
    assert(fineGGraph != coarseGGraph);
  }

  template<typename Context>
  void goSort(GNode node, Context& lwl) {
    //    std::cout << 'p';
    //fineGGraph is read only in this loop, so skip locks
    MetisNode &nodeData = coarseGGraph->getData(node, Galois::MethodFlag::NONE);

    typedef std::deque<std::pair<GNode, unsigned>, Galois::PerIterAllocTy::rebind<std::pair<GNode,unsigned> >::other> GD;
    //copy and translate all edges
    GD edges(GD::allocator_type(lwl.getPerIterAlloc()));

    for (unsigned x = 0; x < nodeData.numChildren(); ++x)
      for (auto ii = fineGGraph->edge_begin(nodeData.getChild(x), Galois::MethodFlag::NONE), ee = fineGGraph->edge_end(nodeData.getChild(x)); ii != ee; ++ii) {
        GNode dst = fineGGraph->getEdgeDst(ii, Galois::MethodFlag::NONE);
        GNode p = fineGGraph->getData(dst, Galois::MethodFlag::NONE).getParent();
        edges.emplace_back(p, fineGGraph->getEdgeData(ii, Galois::MethodFlag::NONE));
      }
    
    //slightly faster not ordering by edge weight
    std::sort(edges.begin(), edges.end(), [] (const std::pair<GNode, unsigned>& lhs, const std::pair<GNode, unsigned>& rhs) { return lhs.first < rhs.first; } );

    //insert edges
    for (auto pp = edges.begin(), ep = edges.end(); pp != ep;) {
      GNode dst = pp->first;
      unsigned sum = pp->second;
      ++pp;
      if (node != dst) { //no self edges
        while (pp != ep && pp->first == dst) {
          sum += pp->second;
          ++pp;
        }
        coarseGGraph->addEdgeWithoutCheck(node, dst, Galois::MethodFlag::NONE, sum);
      }
    }
    //    assert(e);
    //nodeData.setNumEdges(e);
  }

  template<typename Context>
  void operator()(GNode node, Context& lwl) {
    // MetisNode &nodeData = coarseGGraph->getData(node, Galois::MethodFlag::NONE);
    // if (std::distance(fineGGraph->edge_begin(nodeData.getChild(0), Galois::MethodFlag::NONE),
    //                   fineGGraph->edge_begin(nodeData.getChild(0), Galois::MethodFlag::NONE))
    //     < 256)
    //   goSort(node,lwl);
    // else
    //   goHM(node,lwl);
    goSort(node, lwl);
    //goHeap(node,lwl);
  }
};

struct HighDegreeIndexer: public std::unary_function<GNode, unsigned int> {
  static GGraph* indexgraph;
  unsigned int operator()(const GNode& val) const {
    return indexgraph->getData(val, Galois::MethodFlag::NONE).isFailedMatch() ?
      std::numeric_limits<unsigned int>::max() :
      (std::numeric_limits<unsigned int>::max() - 
       ((std::distance(indexgraph->edge_begin(val, Galois::MethodFlag::NONE), 
                       indexgraph->edge_end(val, Galois::MethodFlag::NONE))) >> 2));
  }
};

GGraph* HighDegreeIndexer::indexgraph = 0;

struct LowDegreeIndexer: public std::unary_function<GNode, unsigned int> {
  unsigned int operator()(const GNode& val) const {
    return  HighDegreeIndexer::indexgraph->getData(val, Galois::MethodFlag::NONE).isFailedMatch() ? std::numeric_limits<unsigned int>::max():
      (std::distance(HighDegreeIndexer::indexgraph->edge_begin(val, Galois::MethodFlag::NONE), HighDegreeIndexer::indexgraph->edge_end(val, Galois::MethodFlag::NONE)) >> 2);
  }
};

struct WeightIndexer: public std::unary_function<GNode, int> {
  int operator()(const GNode& val) const {
    return HighDegreeIndexer::indexgraph->getData(val, Galois::MethodFlag::NONE).getWeight();
  }
};

unsigned minRuns(unsigned coarsenTo, unsigned size) {
  unsigned num = 0;
  while (coarsenTo < size) {
    ++num;
    size /= 2;
  }
  return num;
}

void findMatching(MetisGraph* coarseMetisGraph, unsigned iterNum, bool useRM = false, bool with2Hop = false) {
  MetisGraph* fineMetisGraph = coarseMetisGraph->getFinerGraph();

  /*
   * Different worklist versions tried, dChunkedFIFO 256 works best with LC_MORPH_graph.
   * Another good type would be Lazy Iter.
   */
  //typedef Galois::WorkList::ChunkedLIFO<64, GNode> WL;
  //typedef Galois::WorkList::LazyIter<decltype(fineGGraph->local_begin()),false> WL;

  if(useRM) {
    typedef decltype(fineMetisGraph->getGraph()->local_begin()) ITY;
    typedef Galois::WorkList::StableIterator<ITY, true> WL;
    Galois::InsertBag<GNode> bagNoNeighbor;
    parallelMatchAndCreateNodes<RMmatch> pRM(coarseMetisGraph, bagNoNeighbor, with2Hop);
    std::ostringstream name;
    name << "RM_Match_" << iterNum;
    Galois::for_each_local<WL>(*fineMetisGraph->getGraph(), pRM, "match"); /* name.str().c_str());*/
    if(!bagNoNeighbor.empty())
      matchNoNeighborNodes(coarseMetisGraph, bagNoNeighbor);
  } else {
    //FIXME: use obim for SHEM matching
    typedef decltype(fineMetisGraph->getGraph()->local_begin()) ITY;
    typedef Galois::WorkList::StableIterator<ITY, true> WL;
    typedef Galois::WorkList::dChunkedLIFO<16> Chunk;
    typedef Galois::WorkList::OrderedByIntegerMetric<WeightIndexer, Chunk> pW;
    typedef Galois::WorkList::OrderedByIntegerMetric<LowDegreeIndexer, Chunk> pLD;
    typedef Galois::WorkList::OrderedByIntegerMetric<HighDegreeIndexer, Chunk> pHD;

    HighDegreeIndexer::indexgraph = fineMetisGraph->getGraph();

    Galois::InsertBag<GNode> bagNoNeighbor;
    parallelMatchAndCreateNodes<HEMmatch> pHEM(coarseMetisGraph, bagNoNeighbor, with2Hop);
    std::ostringstream name;
     name << "HEM_Match_" << iterNum;
     Galois::for_each_local<pLD>(*fineMetisGraph->getGraph(), pHEM, "match"); /* name.str().c_str()); */
  
    if(!bagNoNeighbor.empty()){
      matchNoNeighborNodes(coarseMetisGraph, bagNoNeighbor);
    }
    //FIXME: decide if we should match null-edge nodes here
  }
}

void createCoarseEdges(MetisGraph *coarseMetisGraph, unsigned iterNum) {
  MetisGraph* fineMetisGraph = coarseMetisGraph->getFinerGraph();
  GGraph* fineGGraph = fineMetisGraph->getGraph();
  typedef Galois::WorkList::StableIterator<decltype(fineGGraph->local_begin()), true> WL;
  parallelPopulateEdges pPE(coarseMetisGraph);
  std::ostringstream name;
  name << "Populate_Edges_" << iterNum;
  Galois::for_each_local<WL>(*coarseMetisGraph->getGraph(), pPE, "popedge");//name.str().c_str());
}

MetisGraph* coarsenOnce(MetisGraph *fineMetisGraph, unsigned iterNum, bool useRM = false, bool with2Hop = false) {
  MetisGraph *coarseMetisGraph = new MetisGraph(fineMetisGraph);
  //assertNoMatched(fineMetisGraph->getGraph());
  //  Galois::Timer t;
  //  t.start();
  findMatching(coarseMetisGraph, iterNum, useRM, with2Hop);
  //  t.stop();
  //  std::cout << "Time Matching " << iterNum << " is " << t.get() << "\n";
  //assertNoMatched(coarseMetisGraph->getGraph());
  //Galois::Timer t2;
  //t2.start();
  createCoarseEdges(coarseMetisGraph, iterNum);
  //t2.stop();
  //std::cout << "Time Creating " << iterNum << " is " << t2.get() << "\n";
  return coarseMetisGraph;
}

} // anon namespace

MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo) {
  unsigned iRuns = minRuns(coarsenTo, std::distance(fineMetisGraph->getGraph()->begin(), fineMetisGraph->getGraph()->end()));
  
  //std::cout << "coarsening stating, now " <<  std::distance(fineMetisGraph->getGraph()->begin(), fineMetisGraph->getGraph()->end()) << " on " << fineMetisGraph << "\n";
  //graphStat(fineMetisGraph->getGraph());
  MetisGraph* coarseGraph = fineMetisGraph;
  unsigned iterNum = 0;
  bool with2Hop = false; 
  while (iRuns) {//overflow
    coarseGraph = coarsenOnce(coarseGraph, iterNum, iterNum <2, with2Hop); 
    //std::cout << "coarsening " << iterNum << " done, now " <<  std::distance(coarseGraph->getGraph()->begin(), coarseGraph->getGraph()->end()) << " on " << coarseGraph << " targeting " << coarsenTo << "\n";
    //    graphStat(coarseGraph->getGraph());
    if (iRuns)
      --iRuns;
    if (!iRuns){
      iRuns = minRuns(coarsenTo, std::distance(coarseGraph->getGraph()->begin(), coarseGraph->getGraph()->end()));
      if (iRuns > iterNum/2) with2Hop = true;
    }
    ++iterNum;
  }
  
  return coarseGraph;
}
