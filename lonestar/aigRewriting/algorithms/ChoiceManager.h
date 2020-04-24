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
 Parallel AIG Choice Insertion December 6, 2018.

*/

#ifndef CHOICEMANAGER_H_
#define CHOICEMANAGER_H_

#include "Aig.h"
#include "CutManager.h"
#include "NPNManager.h"
#include "PreCompGraphManager.h"
#include "galois/worklists/Chunk.h"

#include <vector>

namespace algorithm {

typedef struct ThreadLocalDataCH_ {
  std::vector<bool> currentFaninsPol;
  std::vector<aig::GNode> currentFanins;
  std::vector<aig::GNode> decNodeFunc;

  ThreadLocalDataCH_() : currentFaninsPol(4), currentFanins(4), decNodeFunc(20) { }

} ThreadLocalDataCH;

typedef galois::substrate::PerThreadStorage<ThreadLocalDataCH> PerThreadDataCH;

class ChoiceManager {

private:
  aig::Aig& aig;
  CutManager& cutMan;
  NPNManager& npnMan;
  PreCompGraphManager& pcgMan;
  PerThreadDataCH perThreadDataCH;
  int nFuncs;
  int nGraphs;
	int nChoices;

  bool updateAig(ThreadLocalDataCH* thData, aig::GNode rootNode, aig::NodeData& rootData, DecGraph* decGraph, bool isOutputCompl);
  bool decGraphToAigTry(ThreadLocalDataCH* thData, DecGraph* decGraph);
  aig::GNode decGraphToAigCreate(ThreadLocalDataCH* thData, DecGraph* decGraph);

  void lockFaninCone(aig::Graph& aigGraph, aig::GNode node, Cut* cut);

public:

  ChoiceManager(aig::Aig& aig, CutManager& cutMan, NPNManager& npnMan, PreCompGraphManager& pcgMan, int nGraphs, int nChoinces);

  ~ChoiceManager();

  void createNodeChoices(ThreadLocalDataCH* thData, aig::GNode node);

  aig::Aig& getAig();
  CutManager& getCutMan();
  NPNManager& getNPNMan();
  PreCompGraphManager& getPcgMan();
  PerThreadDataCH& getPerThreadDataCH();
};

void runChoiceOperator(ChoiceManager& chMan);

} /* namespace algorithm */

#endif /* CHOICEMANAGER_H_ */

