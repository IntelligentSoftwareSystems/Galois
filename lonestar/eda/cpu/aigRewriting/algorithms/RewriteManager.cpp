/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "RewriteManager.h"

#include "galois/worklists/Chunk.h"

//#include "galois/runtime/profile.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>

using namespace std::chrono;

namespace algorithm {

RewriteManager::RewriteManager(aig::Aig& aig, CutManager& cutMan,
                               NPNManager& npnMan, PreCompGraphManager& pcgMan,
                               int triesNGraphs, bool useZeros,
                               bool updateLevel)
    : aig(aig), cutMan(cutMan), npnMan(npnMan), pcgMan(pcgMan),
      perThreadContextData(), triesNGraphs(triesNGraphs), useZeros(useZeros),
      updateLevel(updateLevel) {

  nFuncs = (1 << 16);

  for (int i = 0; i < cutMan.getNThreads(); i++) {
    ThreadContextData* threadCtx = perThreadContextData.getRemote(i);
    threadCtx->threadId          = i;
  }

  rewriteTime = 0;
}

RewriteManager::~RewriteManager() {
  // TODO
}

aig::GNode RewriteManager::rewriteNode(ThreadContextData* threadCtx,
                                       aig::GNode node,
                                       GNodeVector& fanoutNodes) {

  aig::Graph& aigGraph    = this->aig.getGraph();
  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::WRITE);

  // Get the node's cuts
  this->cutMan.computeCutsRecursively(node);
  Cut* cutsBegin = this->cutMan.getNodeCuts()[nodeData.id];
  assert(cutsBegin != nullptr);

  threadCtx->bestCutMFFCIds.clear();
  threadCtx->bestCutMFFCPreservedIds.clear();

  Cut* cut;
  char* perm;
  unsigned phase;
  unsigned truth;
  // unsigned bestTruth = 0;
  bool isOutputCompl = false;
  int requiredLevel  = 0;
  int nodesSaved;
  // int bestNodesSaved;
  int currentGain = -1, bestGain = -1;
  int i;
  DecGraph* currentGraph = nullptr;
  DecGraph* bestGraph    = nullptr;

  // Go through the cuts to lock the fanin conee
  for (cut = cutsBegin; cut != nullptr; cut = cut->nextCut) {
    // Consider only 4-input cuts
    if (cut->nLeaves != 4) {
      continue;
    }
    lockFaninCone(aigGraph, node, cut);
  }

  // Go through the cuts to rewrite
  for (cut = cutsBegin; cut != nullptr; cut = cut->nextCut) {

    // Consider only 4-input cuts
    if (cut->nLeaves != 4) {
      continue;
    }

    // Get the fanin permutation
    truth = 0xFFFF & (*(this->cutMan.readTruth(cut)));
    perm  = this->npnMan.getPerms4()[(int)this->npnMan.getPerms()[truth]];
    phase = this->npnMan.getPhases()[truth];

    // Collect fanins with the corresponding permutation/phase
    for (i = 0; i < cut->nLeaves; i++) {
      aig::GNode faninNode = this->aig.getNodes()[cut->leaves[(int)perm[i]]];
      if (faninNode == nullptr) {
        break;
      }
      threadCtx->currentFanins[i]    = faninNode;
      threadCtx->currentFaninsPol[i] = !((phase & (1 << i)) > 0);
    }

    if (i != cut->nLeaves) {
      continue;
    }

    int counter = 0;
    for (aig::GNode faninNode : threadCtx->currentFanins) {
      aig::NodeData& faninNodeData =
          aigGraph.getData(faninNode, galois::MethodFlag::READ);
      if (faninNodeData.nFanout == 1) {
        counter++;
      }
    }

    if (counter > 2) {
      continue;
    }

    // lockFaninCone( aigGraph, node, cut );

    // mark the fanin boundary
    for (aig::GNode faninNode : threadCtx->currentFanins) {
      aig::NodeData& faninNodeData =
          aigGraph.getData(faninNode, galois::MethodFlag::WRITE);
      faninNodeData.nFanout++;
    }

    // label MFFC with current ThreadId and the ThreadTravId
    threadCtx->travId += 1;
    nodesSaved =
        labelMFFC(threadCtx, node, threadCtx->threadId, threadCtx->travId);

    // unmark the fanin boundary
    for (aig::GNode faninNode : threadCtx->currentFanins) {
      aig::NodeData& faninNodeData =
          aigGraph.getData(faninNode, galois::MethodFlag::WRITE);
      faninNodeData.nFanout--;
    }

    // evaluate the cut
    currentGraph = evaluateCut(threadCtx, node, cut, nodesSaved, requiredLevel,
                               currentGain);

    // cheeck if the cut is better than the current best one
    if ((currentGraph != nullptr) && (bestGain < currentGain)) {
      bestGain      = currentGain;
      bestGraph     = currentGraph;
      isOutputCompl = ((phase & (1 << 4)) > 0);
      // bestTruth = 0xFFFF & *this->cutMan.readTruth( cut );
      // bestNodesSaved = nodesSaved;
      // collect fanins in the
      for (size_t i = 0; i < threadCtx->currentFanins.size(); i++) {
        threadCtx->bestFanins[i]    = threadCtx->currentFanins[i];
        threadCtx->bestFaninsPol[i] = threadCtx->currentFaninsPol[i];
      }
      threadCtx->bestCutMFFCIds = threadCtx->currentCutMFFCIds;
      threadCtx->bestCutMFFCPreservedIds =
          threadCtx->currentCutMFFCPreservedIds;
    }
  }

  if (!(bestGain > 0 || (bestGain == 0 && useZeros))) {
    return nullptr;
  }

  assert(bestGraph != nullptr);

  // Preparing structure/AIG tracking for updating the AIG
  for (int j = 0; j < 20; j++) {
    if (j < 4) {
      threadCtx->decNodeFunc[j] =
          threadCtx->bestFanins[j]; // Link cut leaves with the best
                                    // decomposition graph
    } else {
      threadCtx->decNodeFunc[j] = nullptr; // Clear the link table, after leaves
    }
  }

  // Define the MFFC available IDs to be reused
  for (int id : threadCtx->bestCutMFFCPreservedIds) {
    threadCtx->bestCutMFFCIds.erase(id);
  }

  // std::cout << threadCtx->threadId << " - Updating AIG with gain " <<
  // bestGain << std::endl;
  aig::GNode newRoot =
      updateAig(threadCtx, node, bestGraph, fanoutNodes, isOutputCompl);
  // std::cout << threadCtx->threadId << " - Update done " << std::endl;

  return newRoot;
}

void RewriteManager::lockFaninCone(aig::Graph& aigGraph, aig::GNode node,
                                   Cut* cut) {

  aig::NodeData& nodeData =
      aigGraph.getData(node, galois::MethodFlag::READ); // lock

  // If node is a cut leaf
  if ((nodeData.id == cut->leaves[0]) || (nodeData.id == cut->leaves[1]) ||
      (nodeData.id == cut->leaves[2]) || (nodeData.id == cut->leaves[3])) {
    return;
  }

  // If node is a PI
  if ((nodeData.type == aig::NodeType::PI) ||
      (nodeData.type == aig::NodeType::LATCH)) {
    return;
  }

  auto inEdgeIt      = aigGraph.in_edge_begin(node);
  aig::GNode lhsNode = aigGraph.getEdgeDst(inEdgeIt);
  //  aig::NodeData& lhsData =
  aigGraph.getData(lhsNode, galois::MethodFlag::READ); // lock
  inEdgeIt++;
  aig::GNode rhsNode = aigGraph.getEdgeDst(inEdgeIt);
  //  aig::NodeData& rhsData =
  aigGraph.getData(rhsNode, galois::MethodFlag::READ); // lock

  lockFaninCone(aigGraph, lhsNode, cut);
  lockFaninCone(aigGraph, rhsNode, cut);
}

int RewriteManager::labelMFFC(ThreadContextData* threadCtx, aig::GNode node,
                              int threadId, int travId) {

  aig::Graph& aigGraph = this->aig.getGraph();

  threadCtx->currentCutMFFCIds.clear();

  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);
  if ((nodeData.type == aig::NodeType::PI) ||
      (nodeData.type == aig::NodeType::LATCH)) {
    return 0;
  }

  threadCtx->currentCutMFFCIds.insert(nodeData.id);

  int nConeSize1 = refDerefMFFCNodes(threadCtx, node, threadId, travId, false,
                                     true); // dereference
  int GALOIS_USED_ONLY_IN_DEBUG(nConeSize2) =
      refDerefMFFCNodes(threadCtx, node, threadId, travId, true,
                        false); // reference

  assert(nConeSize1 == nConeSize2);
  assert(nConeSize1 > 0);

  return nConeSize1;
}

int RewriteManager::refDerefMFFCNodes(ThreadContextData* threadCtx,
                                      aig::GNode node, int threadId, int travId,
                                      bool reference, bool label) {

  aig::Graph& aigGraph    = this->aig.getGraph();
  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

  // label visited nodes
  if (label) {
    this->aig.registerTravId(nodeData.id, threadId, travId);
  }
  // skip the CI
  if ((nodeData.type == aig::NodeType::PI) ||
      (nodeData.type == aig::NodeType::LATCH)) {
    return 0;
  }

  // process the internal node
  auto inEdgeIt          = aigGraph.in_edge_begin(node);
  aig::GNode lhsNode     = aigGraph.getEdgeDst(inEdgeIt);
  aig::NodeData& lhsData = aigGraph.getData(lhsNode, galois::MethodFlag::WRITE);

  inEdgeIt++;
  aig::GNode rhsNode     = aigGraph.getEdgeDst(inEdgeIt);
  aig::NodeData& rhsData = aigGraph.getData(rhsNode, galois::MethodFlag::WRITE);

  int counter = 1;

  if (reference) {
    if (lhsData.nFanout++ == 0) {
      counter += refDerefMFFCNodes(threadCtx, lhsNode, threadId, travId,
                                   reference, label);
    }
    if (rhsData.nFanout++ == 0) {
      counter += refDerefMFFCNodes(threadCtx, rhsNode, threadId, travId,
                                   reference, label);
    }
  } else {
    assert(lhsData.nFanout > 0);
    assert(rhsData.nFanout > 0);
    if (--lhsData.nFanout == 0) {
      threadCtx->currentCutMFFCIds.insert(lhsData.id);
      counter += refDerefMFFCNodes(threadCtx, lhsNode, threadId, travId,
                                   reference, label);
    }
    if (--rhsData.nFanout == 0) {
      threadCtx->currentCutMFFCIds.insert(rhsData.id);
      counter += refDerefMFFCNodes(threadCtx, rhsNode, threadId, travId,
                                   reference, label);
    }
  }

  return counter;
}

DecGraph* RewriteManager::evaluateCut(ThreadContextData* threadCtx,
                                      aig::GNode root, Cut* cut,
                                      int nNodesSaved, int maxLevel,
                                      int& bestGain) {

  DecGraph* bestGraph = NULL;
  DecGraph* currentGraph;
  ForestNode* node;
  int nNodesAdded;
  unsigned uTruth;
  bestGain = -1;

  threadCtx->currentCutMFFCPreservedIds.clear();

  // find the matching class of subgraphs
  uTruth = 0xFFFF & *this->cutMan.readTruth(cut);
  std::vector<ForestNode*>& subgraphs =
      this->pcgMan.getClasses()[this->npnMan.getMap()[uTruth]];

  aig::Graph& aigGraph = aig.getGraph();

  // copy the leaves
  for (int i = 0; i < 4; i++) { // each deGraph has eactly 4 inputs (vars).
    aig::GNode fanin = threadCtx->currentFanins[i];
    //    aig::NodeData& faninData =
    aigGraph.getData(fanin, galois::MethodFlag::READ);
    threadCtx->decNodeFunc[i] = fanin;
    // threadCtx->decNodeLevel[i] = faninData.level;
  }

  // Pruning
  int nSubgraphs = subgraphs.size();
  if (nSubgraphs > this->triesNGraphs) {
    nSubgraphs = this->triesNGraphs;
  }

  // determine the best subgrap
  for (int i = 0; i < nSubgraphs; i++) {
    node = subgraphs[i];
    // get the current graph
    currentGraph = (DecGraph*)node->pNext;

    // clear link table, after leaves
    for (int j = 4; j < 20; j++) { // each decGraph has at most 20 nodes.
      threadCtx->decNodeFunc[j] = NULL;
    }

    // detect how many unlabeled nodes will be reused
    nNodesAdded = decGraphToAigCount(threadCtx, root, currentGraph, nNodesSaved,
                                     maxLevel);

    if (nNodesAdded == -1) {
      continue;
    }

    assert(nNodesSaved >= nNodesAdded);

    // count the gain at this node
    if (bestGain < nNodesSaved - nNodesAdded) {
      bestGain  = nNodesSaved - nNodesAdded;
      bestGraph = currentGraph;
      threadCtx->currentCutMFFCPreservedIds =
          threadCtx->currentGraphMFFCPreservedIds;
    }
  }

  if (bestGain == -1) {
    return NULL;
  }

  return bestGraph;
}

/*
 *   Before calling this procedure, AIG nodes should be assigned to DecNodes by
 *   using the threadCtx->decNodeFunc[ DecNode.id ] for each leaf of the
 * decGraph. Returns -1 if the number of nodes and levels exceeded the given
 * limit or the number of levels exceeded the maximum allowed level.
 */
int RewriteManager::decGraphToAigCount(ThreadContextData* threadCtx,
                                       aig::GNode root, DecGraph* decGraph,
                                       int maxNode,
                                       int GALOIS_UNUSED(maxLevel)) {

  DecNode* node;
  DecNode* lhsNode;
  DecNode* rhsNode;
  aig::GNode curAnd;
  aig::GNode lhsAnd;
  aig::GNode rhsAnd;
  bool lhsPol, rhsPol;
  int counter = 0;
  // int newLevel, oldLevel;

  aig::Graph& aigGraph = this->aig.getGraph();

  threadCtx->currentGraphMFFCPreservedIds.clear();

  // check for constant function or a literal
  if (decGraph->isConst() || decGraph->isVar()) {
    return counter;
  }

  // compute the AIG size after adding the internal nodes
  for (int i = decGraph->getLeaveNum();
       (i < decGraph->getNodeNum()) && ((node = decGraph->getNode(i)), 1);
       i++) {

    // get the children of this node
    lhsNode = decGraph->getNode(node->eEdge0.Node);
    rhsNode = decGraph->getNode(node->eEdge1.Node);

    // get the AIG nodes corresponding to the children
    lhsAnd = threadCtx->decNodeFunc[lhsNode->id];
    rhsAnd = threadCtx->decNodeFunc[rhsNode->id];

    // if they are both present, find the resulting node
    if (lhsAnd && rhsAnd) {
      if (lhsNode->id < 4) { // If lhs is a cut leaf
        lhsPol = node->eEdge0.fCompl
                     ? !(threadCtx->currentFaninsPol[lhsNode->id])
                     : threadCtx->currentFaninsPol[lhsNode->id];
      } else {
        lhsPol = node->eEdge0.fCompl ? false : true;
      }

      if (rhsNode->id < 4) { // If rhs is a cut leaf
        rhsPol = node->eEdge1.fCompl
                     ? !(threadCtx->currentFaninsPol[rhsNode->id])
                     : threadCtx->currentFaninsPol[rhsNode->id];
      } else {
        rhsPol = node->eEdge1.fCompl ? false : true;
      }

      curAnd = this->aig.lookupNodeInFanoutMap(lhsAnd, rhsAnd, lhsPol, rhsPol);

      // return -1 if the node is the same as the original root
      if (curAnd == root) {
        return -1;
      }
    } else {
      curAnd = nullptr;
    }

    // count the number of new levels
    // newLevel = 1 + std::max( threadCtx->decNodeLevel[ lhsNode->id ],
    // threadCtx->decNodeLevel[ rhsNode->id ] );

    if (curAnd) {
      aig::NodeData& curAndData =
          aigGraph.getData(curAnd, galois::MethodFlag::READ);
      bool isMFFC = this->aig.lookupTravId(curAndData.id, threadCtx->threadId,
                                           threadCtx->travId);

      if (isMFFC) {
        threadCtx->currentGraphMFFCPreservedIds.insert(curAndData.id);
        // count the number of added nodes
        if (++counter > maxNode) {
          return -1;
        }
      }

      // TODO Implement an Heuristic for levels preservation
      /*
      if ( curAnd == aig.getConstZero() ) {
          newLevel = 0;
      }
      else {
          if ( curAnd == lhsAnd ) {
              aig::NodeData & lhsAndData = aigGraph.getData( lhsAnd,
      galois::MethodFlag::READ ); newLevel = lhsAndData.level;
          }
          else {
              if ( curAnd == rhsAnd ) {
                  aig::NodeData & rhsAndData = aigGraph.getData( rhsAnd,
      galois::MethodFlag::READ ); newLevel = rhsAndData.level;
              }
          }
      }

      oldLevel = curAndData.level;
      //assert( LevelNew == LevelOld );
      */
    } else {
      // count the number of added nodes
      if (++counter > maxNode) {
        return -1;
      }
    }

    // if ( newLevel > maxLevel ) {
    //    return -1;
    //}

    threadCtx->decNodeFunc[node->id] = curAnd;
    // threadCtx->decNodeLevel[ node->id ] = newLevel;
  }

  return counter;
}

aig::GNode RewriteManager::updateAig(ThreadContextData* threadCtx,
                                     aig::GNode oldRoot, DecGraph* decGraph,
                                     GNodeVector& fanoutNodes,
                                     bool isOutputCompl) {

  aig::Graph& aigGraph = this->aig.getGraph();

  // Prepare to delete nodes in the MFFC
  for (int id : threadCtx->bestCutMFFCIds) {
    aig::GNode mffcNode = this->aig.getNodes()[id];
    auto inEdge         = aigGraph.in_edge_begin(mffcNode);

    aig::GNode lhsNode = aigGraph.getEdgeDst(inEdge);
    //    aig::NodeData& lhsNodeData =
    aigGraph.getData(lhsNode, galois::MethodFlag::WRITE);
    bool lhsNodePol = aigGraph.getEdgeData(inEdge);
    inEdge++;
    aig::GNode rhsNode = aigGraph.getEdgeDst(inEdge);
    //    aig::NodeData& rhsNodeData =
    aigGraph.getData(rhsNode, galois::MethodFlag::WRITE);
    bool rhsNodePol = aigGraph.getEdgeData(inEdge);

    this->aig.removeNodeInFanoutMap(mffcNode, lhsNode, rhsNode, lhsNodePol,
                                    rhsNodePol);
    this->aig.getNodes()[id] = nullptr;
    this->aig.getFanoutMap(id).clear();
    this->cutMan.recycleNodeCuts(id);
  }

  bool isDecGraphComplement = isOutputCompl
                                  ? (bool)decGraph->getRootEdge().fCompl ^ 1
                                  : (bool)decGraph->getRootEdge().fCompl;
  aig::GNode newRoot;

  // check for constant function
  if (decGraph->isConst()) {
    newRoot = this->aig.getConstZero();
  } else {
    // check for a literal
    if (decGraph->isVar()) {
      DecNode* decNode = decGraph->getVar();
      isDecGraphComplement =
          isDecGraphComplement ? (!threadCtx->bestFaninsPol[decNode->id]) ^ true
                               : !threadCtx->bestFaninsPol[decNode->id];
      newRoot = threadCtx->decNodeFunc[decNode->id];
    } else {
      newRoot = decGraphToAig(threadCtx, decGraph);
    }
  }

  addNewSubgraph(oldRoot, newRoot, fanoutNodes, isDecGraphComplement);

  deleteOldMFFC(aigGraph, oldRoot);

  return newRoot;
}

/*
 *   Transforms the decomposition graph into the AIG.
 *   Before calling this procedure, AIG nodes for the fanins
 *   should be assigned to threadCtx.decNodeFun[ decNode.id ].
 */
aig::GNode RewriteManager::decGraphToAig(ThreadContextData* threadCtx,
                                         DecGraph* decGraph) {

  DecNode* decNode = nullptr;
  DecNode* lhsNode;
  DecNode* rhsNode;
  aig::GNode curAnd;
  aig::GNode lhsAnd;
  aig::GNode rhsAnd;
  bool lhsAndPol;
  bool rhsAndPol;

  // build the AIG nodes corresponding to the AND gates of the graph
  for (int i = decGraph->getLeaveNum();
       (i < decGraph->getNodeNum()) && ((decNode = decGraph->getNode(i)), 1);
       i++) {

    // get the children of this node
    lhsNode = decGraph->getNode(decNode->eEdge0.Node);
    rhsNode = decGraph->getNode(decNode->eEdge1.Node);

    // get the AIG nodes corresponding to the children
    lhsAnd = threadCtx->decNodeFunc[lhsNode->id];
    rhsAnd = threadCtx->decNodeFunc[rhsNode->id];

    if (lhsNode->id < 4) { // If lhs is a cut leaf
      lhsAndPol = decNode->eEdge0.fCompl
                      ? !(threadCtx->bestFaninsPol[lhsNode->id])
                      : threadCtx->bestFaninsPol[lhsNode->id];
    } else {
      lhsAndPol = decNode->eEdge0.fCompl ? false : true;
    }

    if (rhsNode->id < 4) { // If rhs is a cut leaf
      rhsAndPol = decNode->eEdge1.fCompl
                      ? !(threadCtx->bestFaninsPol[rhsNode->id])
                      : threadCtx->bestFaninsPol[rhsNode->id];
    } else {
      rhsAndPol = decNode->eEdge1.fCompl ? false : true;
    }

    curAnd =
        this->aig.lookupNodeInFanoutMap(lhsAnd, rhsAnd, lhsAndPol, rhsAndPol);

    if (curAnd) {
      threadCtx->decNodeFunc[decNode->id] = curAnd;
    } else {
      threadCtx->decNodeFunc[decNode->id] =
          createAndNode(threadCtx, lhsAnd, rhsAnd, lhsAndPol, rhsAndPol);
    }
  }

  return threadCtx->decNodeFunc[decNode->id];
}

aig::GNode RewriteManager::createAndNode(ThreadContextData* threadCtx,
                                         aig::GNode lhsAnd, aig::GNode rhsAnd,
                                         bool lhsAndPol, bool rhsAndPol) {

  aig::Graph& aigGraph = this->aig.getGraph();
  aig::NodeData& lhsAndData =
      aigGraph.getData(lhsAnd, galois::MethodFlag::READ);
  aig::NodeData& rhsAndData =
      aigGraph.getData(rhsAnd, galois::MethodFlag::READ);

  aig::NodeData newAndData;

  auto idIt =
      threadCtx->bestCutMFFCIds.begin(); // reuse an ID from deleted MFFC
  auto id = (*idIt);
  threadCtx->bestCutMFFCIds.erase(
      idIt); // remove the reused ID from the available IDs set
  assert(id < int(this->aig.getNodes().size()));

  newAndData.id      = id;
  newAndData.type    = aig::NodeType::AND;
  newAndData.level   = 1 + std::max(lhsAndData.level, rhsAndData.level);
  newAndData.counter = 0;

  if (lhsAndData.counter == 3) {
    newAndData.counter += 1;
  }

  if (rhsAndData.counter == 3) {
    newAndData.counter += 1;
  }

  if (newAndData.counter == 2) {
    newAndData.counter += 1;
  }

  aig::GNode newAnd = aigGraph.createNode(newAndData);
  aigGraph.addNode(newAnd);

  aigGraph.getEdgeData(aigGraph.addMultiEdge(
      lhsAnd, newAnd, galois::MethodFlag::WRITE)) = lhsAndPol;
  aigGraph.getEdgeData(aigGraph.addMultiEdge(
      rhsAnd, newAnd, galois::MethodFlag::WRITE)) = rhsAndPol;
  lhsAndData.nFanout++;
  rhsAndData.nFanout++;

  // int faninSize = std::distance( aigGraph.in_edge_begin( newAnd ),
  // aigGraph.in_edge_end( newAnd ) ); assert( faninSize == 2 );

  this->aig.getNodes()[id] = newAnd;
  this->aig.insertNodeInFanoutMap(newAnd, lhsAnd, rhsAnd, lhsAndPol, rhsAndPol);

  return newAnd;
}

void RewriteManager::addNewSubgraph(aig::GNode oldNode, aig::GNode newNode,
                                    GNodeVector& fanoutNodes,
                                    bool isNewRootComplement) {

  int fanoutNodesSize = fanoutNodes.size();

  aig::GNode fanoutNode;
  aig::GNode otherNode;
  bool otherNodePol;
  bool newNodePol;
  bool oldNodePol;

  aig::Graph& aigGraph = this->aig.getGraph();
  aig::NodeData& newNodeData =
      aigGraph.getData(newNode, galois::MethodFlag::READ);
  aig::NodeData& oldNodeData =
      aigGraph.getData(oldNode, galois::MethodFlag::READ);
  assert(oldNodeData.nFanout == fanoutNodesSize);

  // look at the fanouts of old node
  for (int i = 0; i < fanoutNodesSize; i++) {

    fanoutNode = fanoutNodes[i];
    aig::NodeData& fanoutNodeData =
        aigGraph.getData(fanoutNode, galois::MethodFlag::READ);

    // auto outEdge = aigGraph.findEdge( oldNode, fanoutNode );
    auto fanoutNodeInEdge = aigGraph.findInEdge(oldNode, fanoutNode);

    if (fanoutNodeInEdge == aigGraph.in_edge_end(fanoutNode)) {
      std::cout << "Adding new subgraph, fanoutNode inEdge not found!"
                << std::endl;
    }

    oldNodePol = aigGraph.getEdgeData(fanoutNodeInEdge);
    // newNodePol = isNewRootComplement ? !(false ^ oldNodePol) : !(true ^
    // oldNodePol);
    newNodePol = isNewRootComplement ? !(oldNodePol) : oldNodePol;

    if ((fanoutNodeData.type == aig::NodeType::PO) ||
        (fanoutNodeData.type == aig::NodeType::LATCH)) {
      // remove the oldNode from the fanoutNode's fanin
      // aigGraph.removeEdge( oldNode, fanoutEdge );
      aigGraph.removeInEdge(fanoutNode, fanoutNodeInEdge);
      oldNodeData.nFanout--;
      // add newNode to the fanoutNode's fanin
      aigGraph.getEdgeData(aigGraph.addMultiEdge(
          newNode, fanoutNode, galois::MethodFlag::WRITE)) = newNodePol;
      newNodeData.nFanout++;
      fanoutNodeData.level = newNodeData.level;
      continue;
    }

    // find the otherNode diffetent of oldNode as a fanin of the fanoutNode
    auto inEdge  = aigGraph.in_edge_begin(fanoutNode);
    otherNode    = aigGraph.getEdgeDst(inEdge);
    otherNodePol = aigGraph.getEdgeData(inEdge);

    if (otherNode == oldNode) {
      inEdge++;
      otherNode    = aigGraph.getEdgeDst(inEdge);
      otherNodePol = aigGraph.getEdgeData(inEdge);
    }

    assert(newNode != otherNode);

    // Remove fanoutNode from the fanoutMap from otherNode
    this->aig.removeNodeInFanoutMap(fanoutNode, otherNode, oldNode,
                                    otherNodePol, oldNodePol);

    // remove the oldNode from the fanoutNode fanin
    // aigGraph.removeEdge( oldNode, fanoutEdge );
    aigGraph.removeInEdge(fanoutNode, fanoutNodeInEdge);
    oldNodeData.nFanout--;

    // add newNode to the fanoutNode fanins
    aigGraph.getEdgeData(aigGraph.addMultiEdge(
        newNode, fanoutNode, galois::MethodFlag::WRITE)) = newNodePol;
    newNodeData.nFanout++;

    aig::NodeData& otherNodeData =
        aigGraph.getData(otherNode, galois::MethodFlag::READ);
    fanoutNodeData.level = 1 + std::max(newNodeData.level, otherNodeData.level);

    // Insert fanoutNode in the fanoutMap from other Node with new inEdge
    this->aig.insertNodeInFanoutMap(fanoutNode, otherNode, newNode,
                                    otherNodePol, newNodePol);
  }
}

void RewriteManager::deleteOldMFFC(aig::Graph& aigGraph, aig::GNode oldNode) {

  // assert( oldNode != nullptr );

  aig::NodeData& oldNodeData =
      aigGraph.getData(oldNode, galois::MethodFlag::READ);

  if ((oldNodeData.type == aig::NodeType::AND) && (oldNodeData.nFanout == 0) &&
      aigGraph.containsNode(oldNode, galois::MethodFlag::WRITE)) {
    deleteOldMFFCRec(aigGraph, oldNode);
  }
}

void RewriteManager::deleteOldMFFCRec(aig::Graph& aigGraph,
                                      aig::GNode oldNode) {

  auto inEdge        = aigGraph.in_edge_begin(oldNode);
  aig::GNode lhsNode = aigGraph.getEdgeDst(inEdge);
  aig::NodeData& lhsNodeData =
      aigGraph.getData(lhsNode, galois::MethodFlag::WRITE);
  inEdge++;
  aig::GNode rhsNode = aigGraph.getEdgeDst(inEdge);
  aig::NodeData& rhsNodeData =
      aigGraph.getData(rhsNode, galois::MethodFlag::WRITE);

  // assert( (lhsNode != nullptr) && (rhsNode != nullptr) );

  aigGraph.removeNode(oldNode);
  lhsNodeData.nFanout--;
  rhsNodeData.nFanout--;

  if ((lhsNodeData.type == aig::NodeType::AND) && (lhsNodeData.nFanout == 0) &&
      aigGraph.containsNode(lhsNode, galois::MethodFlag::WRITE)) {
    deleteOldMFFCRec(aigGraph, lhsNode);
  }

  if ((rhsNodeData.type == aig::NodeType::AND) && (rhsNodeData.nFanout == 0) &&
      aigGraph.containsNode(rhsNode, galois::MethodFlag::WRITE)) {
    deleteOldMFFCRec(aigGraph, rhsNode);
  }
}

aig::Aig& RewriteManager::getAig() { return this->aig; }

CutManager& RewriteManager::getCutMan() { return this->cutMan; }

NPNManager& RewriteManager::getNPNMan() { return this->npnMan; }

PreCompGraphManager& RewriteManager::getPcgMan() { return this->pcgMan; }

PerThreadContextData& RewriteManager::getPerThreadContextData() {
  return this->perThreadContextData;
}

bool RewriteManager::getUseZerosFlag() { return this->useZeros; }

bool RewriteManager::getUpdateLevelFlag() { return this->updateLevel; }

long double RewriteManager::getRewriteTime() { return this->rewriteTime; }

void RewriteManager::setRewriteTime(long double time) {
  this->rewriteTime = time;
}

struct RewriteOperator {

  RewriteManager& rwtMan;
  CutManager& cutMan;

  RewriteOperator(RewriteManager& rwtMan)
      : rwtMan(rwtMan), cutMan(rwtMan.getCutMan()) {}

  void operator()(aig::GNode node, galois::UserContext<aig::GNode>& ctx) {

    aig::Graph& aigGraph = rwtMan.getAig().getGraph();

    if ((node == nullptr) ||
        !aigGraph.containsNode(node, galois::MethodFlag::WRITE)) {
      return;
    }

    aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::WRITE);

    if (nodeData.type == aig::NodeType::AND) {

      if ((nodeData.nFanout < 1000)) {

        Alloc& alloc = ctx.getPerIterAlloc();
        GNodeVector fanoutNodes(alloc);

        // Touching outgoing neighobors to acquire their locks and their fanin
        // node's locks.
        for (auto outEdge : aigGraph.out_edges(node)) {
          aig::GNode fanoutNode = aigGraph.getEdgeDst(outEdge);
          fanoutNodes.push_back(fanoutNode);
          aigGraph.in_edges(fanoutNode);
        }

        ThreadContextData* threadCtx =
            rwtMan.getPerThreadContextData().getLocal();

        // Try to rewrite the node
        aig::GNode newNode = rwtMan.rewriteNode(threadCtx, node, fanoutNodes);

        bool scheduleFanoutNodes = false;

        if (newNode == nullptr) { // it means that node was not rewritten
          if (nodeData.counter == 2) {
            nodeData.counter += 1;
          }

          if (nodeData.counter == 3) {
            scheduleFanoutNodes = true;
          }
        } else {
          aig::NodeData& newNodeData =
              aigGraph.getData(newNode, galois::MethodFlag::READ);
          if (newNodeData.counter == 3) {
            scheduleFanoutNodes = true;
          }
        }

        if (scheduleFanoutNodes) {
          for (aig::GNode nextNode : fanoutNodes) {
            aig::NodeData& nextNodeData =
                aigGraph.getData(nextNode, galois::MethodFlag::WRITE);

            if ((nextNodeData.type == aig::NodeType::PO) ||
                (nextNodeData.type == aig::NodeType::LATCH)) {
              continue;
            }

            nextNodeData.counter += 1;
            if (nextNodeData.counter == 2) {
              if (cutMan.getNodeCuts()[nextNodeData.id] != nullptr) {
                cutMan.recycleNodeCuts(nextNodeData.id);
              }
              // rwtMan.nPushes += 1;
              ctx.push(nextNode);
            }
          }
        }

      } else {

        // Touching outgoing neighobors to acquire their locks and their fanin
        // node's locks.
        aigGraph.out_edges(node);

        if (nodeData.counter == 2) {
          nodeData.counter += 1;
        }

        if (nodeData.counter == 3) {
          // Insert nextNodes in the worklist
          for (auto outEdge : aigGraph.out_edges(node)) {
            aig::GNode nextNode = aigGraph.getEdgeDst(outEdge);
            aig::NodeData& nextNodeData =
                aigGraph.getData(nextNode, galois::MethodFlag::WRITE);

            if ((nextNodeData.type == aig::NodeType::PO) ||
                (nextNodeData.type == aig::NodeType::LATCH)) {
              continue;
            }

            nextNodeData.counter += 1;
            if (nextNodeData.counter == 2) {
              if (cutMan.getNodeCuts()[nextNodeData.id] != nullptr) {
                cutMan.recycleNodeCuts(nextNodeData.id);
              }
              // rwtMan.nPushes += 1;
              ctx.push(nextNode);
            }
          }
        }
      }
    } else {
      if ((nodeData.type == aig::NodeType::PI) ||
          (nodeData.type == aig::NodeType::LATCH)) {

        // Touching outgoing neighobors to acquire their locks and their fanin
        // node's locks.
        aigGraph.out_edges(node);

        // Set the trivial cut
        nodeData.counter      = 3;
        CutPool* cutPool      = cutMan.getPerThreadCutPool().getLocal();
        Cut* trivialCut       = cutPool->getMemory();
        trivialCut->leaves[0] = nodeData.id;
        trivialCut->nLeaves++;
        trivialCut->sig = (1U << (nodeData.id % 31));
        if (cutMan.getCompTruthFlag()) {
          unsigned* cutTruth = cutMan.readTruth(trivialCut);
          for (int i = 0; i < cutMan.getNWords(); i++) {
            cutTruth[i] = 0xAAAAAAAA;
          }
        }
        cutMan.getNodeCuts()[nodeData.id] = trivialCut;

        // Schedule next nodes
        for (auto edge : aigGraph.out_edges(node)) {
          aig::GNode nextNode = aigGraph.getEdgeDst(edge);
          aig::NodeData& nextNodeData =
              aigGraph.getData(nextNode, galois::MethodFlag::WRITE);

          if ((nextNodeData.type == aig::NodeType::PO) ||
              (nextNodeData.type == aig::NodeType::LATCH)) {
            continue;
          }

          nextNodeData.counter += 1;
          if (nextNodeData.counter == 2) {
            // rwtMan.nPushes += 1;
            ctx.push(nextNode);
          }
        }
      }
    }
  }
};

void runRewriteOperator(RewriteManager& rwtMan) {

  // galois::runtime::profileVtune(

  galois::InsertBag<aig::GNode> workList;
  typedef galois::worklists::PerSocketChunkBag<500> DC_BAG;
  // typedef galois::worklists::PerSocketChunkFIFO< 5000 > DC_FIFO;
  // typedef galois::worklists::PerSocketChunkLIFO< 5000 > DC_LIFO;
  // typedef galois::worklists::PerThreadChunkFIFO< 5000 > AC_FIFO;

  for (auto pi : rwtMan.getAig().getInputNodes()) {
    workList.push(pi);
  }

  for (auto latch : rwtMan.getAig().getLatchNodes()) {
    workList.push(latch);
  }

  // Galois Parallel Foreach
  galois::for_each(galois::iterate(workList.begin(), workList.end()),
                   RewriteOperator(rwtMan), galois::wl<DC_BAG>(),
                   galois::loopname("RewriteOperator"),
                   galois::per_iter_alloc());

  // galois::wl<galois::worklists::Deterministic<>>(),
  // galois::wl<DC_BAG>(),

  //,"REWRITING" );
}

} /* namespace algorithm */
