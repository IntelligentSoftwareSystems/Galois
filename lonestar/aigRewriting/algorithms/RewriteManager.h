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

#ifndef REWRITEMANAGER_H_
#define REWRITEMANAGER_H_

#include "Aig.h"
#include "CutManager.h"
#include "NPNManager.h"
#include "PreCompGraphManager.h"

#include "galois/worklists/Chunk.h"

#include <vector>
#include <unordered_set>

namespace algorithm {

typedef struct ThreadContextData_ {
  // Labels
  int threadId;
  int travId;
  // Cut under evaluation data
  std::vector<bool> currentFaninsPol;
  std::vector<bool> bestFaninsPol;
  std::vector<aig::GNode> currentFanins;
  std::vector<aig::GNode> bestFanins;
  // Decomposition graphs data
  std::vector<aig::GNode> decNodeFunc;
  // std::vector< int > decNodeLevel;
  // MFFC IDs to be reused
  std::unordered_set<int> currentCutMFFCIds;
  std::unordered_set<int> bestCutMFFCIds;
  std::unordered_set<int> currentGraphMFFCPreservedIds;
  std::unordered_set<int> currentCutMFFCPreservedIds;
  std::unordered_set<int> bestCutMFFCPreservedIds;

  ThreadContextData_()
      : threadId(0), travId(0), currentFaninsPol(4), bestFaninsPol(4),
        currentFanins(4), bestFanins(4), decNodeFunc(20) {
  } //, decNodeLevel( 20 ) { }

} ThreadContextData;

typedef galois::PerIterAllocTy Alloc;
typedef std::vector<int, galois::PerIterAllocTy::rebind<int>::other> IntVector;
typedef std::vector<aig::GNode, galois::PerIterAllocTy::rebind<aig::GNode>::other> GNodeVector;
typedef std::unordered_set<int, std::hash<int>, std::equal_to<int>, galois::PerIterAllocTy::rebind<int>::other> IntSet;

typedef galois::substrate::PerThreadStorage<ThreadContextData> PerThreadContextData;

class RewriteManager {

private:
  aig::Aig& aig;
  CutManager& cutMan;
  NPNManager& npnMan;
  PreCompGraphManager& pcgMan;

  PerThreadContextData perThreadContextData;

  int nFuncs;
  int triesNGraphs;
  bool useZeros;
  bool updateLevel;

  long double rewriteTime;

  void lockFaninCone(aig::Graph& aigGraph, aig::GNode node, Cut* cut);
  int labelMFFC(ThreadContextData* threadCtx, aig::GNode node, int threadId,
                int travId);
  int refDerefMFFCNodes(ThreadContextData* threadCtx, aig::GNode node,
                        int threadId, int travId, bool reference, bool label);

  DecGraph* evaluateCut(ThreadContextData* threadCtx, aig::GNode root, Cut* cut,
                        int nNodesSaved, int maxLevel, int& bestGain);
  int decGraphToAigCount(ThreadContextData* threadCtx, aig::GNode root,
                         DecGraph* decGraph, int maxNode, int maxLevel);
  aig::GNode updateAig(ThreadContextData* threadCtx, aig::GNode oldRoot,
                       DecGraph* decGraph, GNodeVector& fanoutNodes,
                       bool isOutputCompl, bool updateLevel, int gain);
  aig::GNode decGraphToAig(ThreadContextData* threadCtx, DecGraph* decGraph);
  aig::GNode createAndNode(ThreadContextData* threadCtx, aig::GNode lhsAnd,
                           aig::GNode rhsAnd, bool lhsAndPol, bool rhsAndPol);
  void addNewSubgraph(ThreadContextData* threadCtx, aig::GNode oldNode,
                      aig::GNode newNode, GNodeVector& fanoutNodes,
                      bool isNewRootComplement, bool updateLevel);
  void deleteOldMFFC(aig::Graph& aigGraph, aig::GNode oldNode);
  void deleteOldMFFCRec(aig::Graph& aigGraph, aig::GNode oldNode);

  // void recycleIDsAndCuts( ThreadContextData * threadCtx, IntVector &
  // availableIDs ); aig::GNode searchNode( aig::GNode lhsNode, aig::GNode
  // rhsNode, bool lhsPol, bool rhsPol );

  /*
  void buildLocalStrash( ThreadContextData * threadCtx, Cut * cut, IntSet &
  visited ); void addLocalStrash( ThreadContextData * threadCtx, aig::GNode node
  ); aig::GNode lookupLocalStrash( ThreadContextData * threadCtx, aig::GNode
  lhsNode, aig::GNode rhsNode, bool lhsPol, bool rhsPol ); int makeAndHashKey(
  aig::GNode lhsNode, aig::GNode rhsNode, bool lhsPol, bool rhsPol ); void
  showLocalStrash( std::vector< aig::GNode > & strashMap );
  */

public:
  galois::GAccumulator<long int> nPushes;

  RewriteManager(aig::Aig& aig, CutManager& cutMan, NPNManager& npnMan,
                 PreCompGraphManager& pcgMan, int triesNGraphs, bool useZeros,
                 bool updateLevel);

  ~RewriteManager();

  aig::GNode rewriteNode(ThreadContextData* threadCtx, aig::GNode node,
                         GNodeVector& fanoutNodes);

  aig::Aig& getAig();
  CutManager& getCutMan();
  NPNManager& getNPNMan();
  PreCompGraphManager& getPcgMan();
  PerThreadContextData& getPerThreadContextData();
  bool getUseZerosFlag();
  bool getUpdateLevelFlag();
  long double getRewriteTime();
  void setRewriteTime(long double time);
};

void runRewriteOperator(RewriteManager& rwtMan);

} /* namespace algorithm */

#endif /* REWRITEMANAGER_H_ */
