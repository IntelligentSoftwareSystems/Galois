/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

/*

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#include "ReconvDrivenCut.h"

#include <iostream>
#include <unordered_set>

namespace algorithm {

typedef galois::PerIterAllocTy Alloc;
typedef std::unordered_set<aig::GNode, std::hash<aig::GNode>,
                           std::equal_to<aig::GNode>,
                           galois::PerIterAllocTy::rebind<aig::GNode>::other>
    GNodeSet;

ReconvDrivenCut::ReconvDrivenCut(aig::Aig& aig) : aig(aig) {}

ReconvDrivenCut::~ReconvDrivenCut() {}

struct Preprocess {

  aig::Graph& aigGraph;
  galois::InsertBag<aig::GNode>& workList;

  Preprocess(aig::Graph& aigGraph, galois::InsertBag<aig::GNode>& workList)
      : aigGraph(aigGraph), workList(workList) {}

  void operator()(aig::GNode node) const {

    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);

    if ((nodeData.type == aig::NodeType::AND) && (nodeData.counter == 0) &&
        (nodeData.nFanout < 1000)) {
      workList.push(node);
    }
  }
};

struct ReconvergenceDrivenCut {

  // typedef int tt_does_not_need_aborts;
  // typedef int tt_needs_per_iter_alloc;
  // typedef int tt_does_not_need_push;

  aig::Graph& aigGraph;
  PerThreadRDCutData& perThreadRDCutData;
  size_t cutSizeLimit;

  ReconvergenceDrivenCut(aig::Graph& aigGraph,
                         PerThreadRDCutData& perThreadRDCutData,
                         size_t cutSizeLimit)
      : aigGraph(aigGraph), perThreadRDCutData(perThreadRDCutData),
        cutSizeLimit(cutSizeLimit) {}

  void operator()(aig::GNode node, galois::UserContext<aig::GNode>& ctx) const {
    // void operator()( aig::GNode node ) const {

    aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

    // if ( nodeData.type == aig::NodeType::AND ) {
    if ((nodeData.type == aig::NodeType::AND) && (nodeData.counter == 0) &&
        (nodeData.nFanout < 1000)) {

      // galois::PerIterAllocTy & allocator = ctx.getPerIterAlloc();

      // GNodeSet leaves( allocator );
      // GNodeSet visited( allocator );

      // leaves.insert( node );
      // visited.insert( node );

      RDCutData* rdCutData = perThreadRDCutData.getLocal();

      rdCutData->visited.clear();
      rdCutData->leaves.clear();

      rdCutData->visited.insert(node);
      rdCutData->leaves.insert(node);

      // constructCut( leaves, visited );
      constructCut_iter(rdCutData->leaves, rdCutData->visited);

      /*
      std::cout << "Leaves = { ";
      for ( auto leaf : rdCutData->leaves ) {
          aig::NodeData & leafData = aigGraph.getData( leaf,
      galois::MethodFlag::READ ); std::cout << leafData.id << " ";
      }
      std::cout << "} " << std::endl;

      std::cout << "Visited = { ";
      for ( auto vis : rdCutData->visited ) {
          aig::NodeData & visData = aigGraph.getData( vis,
      galois::MethodFlag::READ ); std::cout << visData.id << " ";
      }
      std::cout << "} " << std::endl;
      */
    }

    nodeData.counter = 1;

    for (auto inEdge : aigGraph.in_edges(node)) {

      aig::GNode inNode = aigGraph.getEdgeDst(inEdge);
      aig::NodeData& inNodeData =
          aigGraph.getData(inNode, galois::MethodFlag::WRITE);

      if ((inNodeData.type == aig::NodeType::AND) &&
          (inNodeData.counter == 0)) {
        ctx.push(inNode);
      }
    }
  }

  /*
      void constructCut( GNodeSet & leaves, GNodeSet & visited ) const {

          aig::GNode minCostNode = nullptr;
          int minCost = std::numeric_limits<int>::max();
          bool onlyPIs = true;
          for ( aig::GNode node : leaves ) {
              aig::NodeData & nodeData = aigGraph.getData( node,
     galois::MethodFlag::READ ); if ( nodeData.type != aig::NodeType::PI ) { int
     cost = leafCost( node, visited ); if ( minCost > cost ) { minCost = cost;
                      minCostNode = node;
                      onlyPIs = false;
                  }
              }
          }
          if ( onlyPIs || (leaves.size() + minCost) > cutSizeLimit ) {
              return;
          }

          if( minCostNode == nullptr ) {
              std::cout << "MinCostNode is null" << std::endl;
              exit( 1 );
          }

          leaves.erase( minCostNode );
          for ( auto edge : aigGraph.in_edges( minCostNode ) ) {
              aig::GNode currentNode = aigGraph.getEdgeDst( edge );
              leaves.insert( currentNode );
              visited.insert( currentNode );
          }

          constructCut( leaves, visited );
      }
  */

  // ITER
  // void constructCut_iter( GNodeSet & leaves, GNodeSet & visited ) const {
  void constructCut_iter(std::unordered_set<aig::GNode>& leaves,
                         std::unordered_set<aig::GNode>& visited) const {

    while (true) {
      aig::GNode minCostNode = nullptr;
      int minCost            = std::numeric_limits<int>::max();
      bool onlyPIs           = true;
      for (aig::GNode node : leaves) {
        aig::NodeData& nodeData =
            aigGraph.getData(node, galois::MethodFlag::READ);
        if (nodeData.type != aig::NodeType::PI) {
          int cost = leafCost(node, visited);
          if (minCost > cost) {
            minCost     = cost;
            minCostNode = node;
            onlyPIs     = false;
          }
        }
      }

      if (onlyPIs || (leaves.size() + minCost) > cutSizeLimit) {
        break;
      }

      if (minCostNode == nullptr) {
        std::cout << "MinCostNode is null" << std::endl;
        exit(1);
      }

      leaves.erase(minCostNode);
      for (auto edge : aigGraph.in_edges(minCostNode)) {
        aig::GNode currentNode = aigGraph.getEdgeDst(edge);
        leaves.insert(currentNode);
        visited.insert(currentNode);
      }
    }
  }

  // int leafCost( aig::GNode & node, GNodeSet & visited ) const {
  int leafCost(aig::GNode& node,
               std::unordered_set<aig::GNode>& visited) const {

    int cost = -1;
    for (auto edge : aigGraph.in_edges(node)) {
      aig::GNode currentNode = aigGraph.getEdgeDst(edge);
      auto it                = visited.find(currentNode);
      if (it == visited.end()) {
        cost++;
      }
    }
    return cost;
  }
};

void ReconvDrivenCut::run(size_t cutSizeLimit) {

  aig::Graph& aigGraph = this->aig.getGraph();

  galois::InsertBag<aig::GNode> workList;
  typedef galois::worklists::PerSocketChunkFIFO<5000> DC_FIFO;

  // typedef galois::worklists::PerSocketChunkBag<5000> DC_BAG;
  // galois::do_all_local( aigGraph, Preprocess( aigGraph, workList ) );
  // galois::for_each_local( workList, ReconvergenceDrivenCut( aigGraph,
  // cutSizeLimit ), galois::wl< DC_BAG >() );

  // galois::for_each( aigGraph.begin(), aigGraph.end(), ReconvergenceDrivenCut(
  // aigGraph, cutSizeLimit ) );

  /*
      for ( aig::GNode po : this->aig.getOutputNodes() ) {
          auto inEdge = aigGraph.in_edge_begin( po );
          aig::GNode inNode = aigGraph.getEdgeDst( inEdge );
          workList.push( inNode );
      }

  */

  /*
      typedef struct FanoutComparator_ {

          aig::Graph & aigGraph;

          FanoutComparator_( aig::Graph & aigGraph ) : aigGraph( aigGraph ) { }

          bool operator()( aig::GNode lhs, aig::GNode rhs ) const {
              aig::NodeData & lhsData = aigGraph.getData( lhs,
     galois::MethodFlag::UNPROTECTED ); aig::NodeData & rhsData =
     aigGraph.getData( rhs, galois::MethodFlag::UNPROTECTED ); return
     lhsData.nFanout > rhsData.nFanout;
          }

      } FanoutComparator;

      std::vector< aig::GNode > nodes = aig.getNodes();

      std::sort( nodes.begin(), nodes.end(), FanoutComparator( aigGraph ) );

      for ( aig::GNode node : nodes ) {
          aig::NodeData & nodeData = aigGraph.getData( node,
     galois::MethodFlag::UNPROTECTED );

          if ( (nodeData.type == aig::NodeType::AND) ) {
              workList.push( node );
          }
      }
  */

  galois::for_each(
      galois::iterate(workList.begin(), workList.end()),
      ReconvergenceDrivenCut(aigGraph, perThreadRDCutData, cutSizeLimit),
      galois::wl<DC_FIFO>(), galois::loopname("ReconvergenceDrivenCut"));
}

} /* namespace algorithm */
