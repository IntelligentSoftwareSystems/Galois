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

#include "ChoiceManager.h"
#include "galois/worklists/Chunk.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>

using namespace std::chrono;

namespace algorithm {

ChoiceManager::ChoiceManager(aig::Aig& aig, CutManager& cutMan, NPNManager& npnMan, PreCompGraphManager& pcgMan, int nGraphs, int nChoices)
    												 : aig(aig), cutMan(cutMan), npnMan(npnMan), pcgMan(pcgMan),
      												 perThreadDataCH(), nGraphs(nGraphs), nChoices(nChoices) {

  nFuncs = (1 << 16);
}

ChoiceManager::~ChoiceManager() { }

void ChoiceManager::createNodeChoices(ThreadLocalDataCH* thData, aig::GNode node) {

  aig::Graph& aigGraph    = this->aig.getGraph();
  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::WRITE);

  // Get the node's cuts
  this->cutMan.computeCutsRecursively(node);
  Cut* cutsBegin = this->cutMan.getNodeCuts()[nodeData.id];
  assert(cutsBegin != nullptr);

  char* perm;
  unsigned phase;
  unsigned truth;
  bool isOutputCompl = false;
  int requiredLevel  = 0;
 	int nSubgraphs;
	int addedChoices = 0;
  int i;
	Cut* cut = nullptr;
	DecGraph* currentGraph = nullptr;
  ForestNode* forestNode = nullptr;

	/*
  // Go through the cuts to lock the fanin conee
  for (cut = cutsBegin; cut != nullptr; cut = cut->nextCut) {
    // Consider only 4-input cuts
    if (cut->nLeaves != 4) {
      continue;
    }
    lockFaninCone(aigGraph, node, cut);
  }
	*/

  // Go through the cuts to rewrite
  for (cut = cutsBegin; cut != nullptr; cut = cut->nextCut) {

		if ( addedChoices >= this->nChoices ) {
			break;
		}

    // Consider only 4-input cuts
    if (cut->nLeaves != 4) {
      continue;
    }

    // Get the fanin permutation
    truth = 0xFFFF & (*(this->cutMan.readTruth(cut)));
    perm  = this->npnMan.getPerms4()[(int)this->npnMan.getPerms()[truth]];
    phase = this->npnMan.getPhases()[truth];
    isOutputCompl = ((phase & (1 << 4)) > 0);

		// Collect fanins with the corresponding permutation/phase
    for (i = 0; i < cut->nLeaves; i++) {
      aig::GNode faninNode = this->aig.getNodes()[cut->leaves[(int)perm[i]]];
      if (faninNode == nullptr) {
        break;
      }
      thData->currentFanins[i]    = faninNode;
      thData->currentFaninsPol[i] = !((phase & (1 << i)) > 0);
    }

    if (i != cut->nLeaves) {
      continue;
    }
   
	 	// find the matching class of subgraphs
		std::vector<ForestNode*>& subgraphs = this->pcgMan.getClasses()[this->npnMan.getMap()[truth]];

		// Pruning
		int nSubgraphs = subgraphs.size();
		if (nSubgraphs > this->nGraphs) {
			nSubgraphs = this->nGraphs;
		}

		// determine the best subgrap
		for (i = 0; i < nSubgraphs; i++) {

			forestNode = subgraphs[i];
			currentGraph = (DecGraph*)forestNode->pNext; // get the current graph

		  bool isComplemented = isOutputCompl ? (bool)currentGraph->getRootEdge().fCompl ^ 1
                                 					: (bool)currentGraph->getRootEdge().fCompl;

			if (isComplemented) {
				continue;
			}		

			// Preparing structure/AIG tracking for updating the AIG 
			for (int j = 0; j < 20; j++) {
				if (j < 4) {
					thData->decNodeFunc[j] =	thData->currentFanins[j]; // Link cut leaves to the decomposition graph
				} else {
					thData->decNodeFunc[j] = nullptr; // Clear the link table, after leaves
				}
			}

			bool wasAdded = updateAig(thData, node, nodeData, currentGraph, isOutputCompl);
		
			if ( wasAdded ) {
				addedChoices++;
			}
		}
	}
}

bool ChoiceManager::updateAig(ThreadLocalDataCH* thData, aig::GNode rootNode, aig::NodeData& rootData, DecGraph* decGraph, bool isOutputCompl) {

  aig::GNode choiceNode;
  aig::GNode auxNode;
  aig::Graph& aigGraph = this->aig.getGraph();

  bool isDecGraphComplemented = isOutputCompl ? (bool)decGraph->getRootEdge().fCompl ^ 1
                                 							: (bool)decGraph->getRootEdge().fCompl;
	
  // check for constant function
  if (decGraph->isConst()) {
    choiceNode = this->aig.getConstZero();
  } 
	else {
    // check for a literal
    if (decGraph->isVar()) {
      DecNode* decNode = decGraph->getVar();
      isDecGraphComplemented = isDecGraphComplemented ? (!thData->currentFaninsPol[decNode->id]) ^ true
                               												: !thData->currentFaninsPol[decNode->id];
      choiceNode = thData->decNodeFunc[decNode->id];
    } 
		else {
			bool isFeasible = decGraphToAigTry(thData, decGraph);
			if ( isFeasible ) {
	      choiceNode = decGraphToAigCreate(thData, decGraph);
			}
			else {
				return false;
			}
    }
  }
	
	if (rootNode == choiceNode) {
		return false;
	}

	aig::NodeData& choiceNodeData = aigGraph.getData(choiceNode, galois::MethodFlag::WRITE);	
	choiceNodeData.choiceList = nullptr;
	//choiceNodeData.isCompl = isDecGraphComplemented;

	aig::GNode currChoice = rootData.choiceList; 

	while ( currChoice != nullptr ) {
		if (choiceNode == currChoice) {
			return false;
		}

		aig::NodeData& currChoiceData = aigGraph.getData(currChoice, galois::MethodFlag::WRITE);
		
		if (currChoiceData.choiceList == nullptr ) {
			currChoiceData.choiceList = choiceNode;
			return true;
		}
		else {
			currChoice = currChoiceData.choiceList;
		}
	}
	
	rootData.choiceList = choiceNode;

	//std::cout << "Node " << choiceNodeData.id << " was added as choice to node " << rootData.id << std::endl;

	return true;
}

/* Transforms the decomposition graph into the AIG. Before calling this procedure, AIG nodes 
 * for the fanins (cut's leaves) should be assigned to thData->decNodeFun[ decNode.id ]. */
bool ChoiceManager::decGraphToAigTry(ThreadLocalDataCH* thData, DecGraph* decGraph) {

  DecNode* decNode;
  DecNode* lhsNode;
  DecNode* rhsNode;
  aig::GNode curAnd;
  aig::GNode lhsAnd;
  aig::GNode rhsAnd;
  bool lhsAndPol;
  bool rhsAndPol;
	aig::Graph& aigGraph = this->aig.getGraph();

  // build the AIG nodes corresponding to the AND gates of the graph
  for (int i = decGraph->getLeaveNum(); (i < decGraph->getNodeNum()) && ((decNode = decGraph->getNode(i)), 1); i++) {

    // get the children of this node
    lhsNode = decGraph->getNode(decNode->eEdge0.Node);
    rhsNode = decGraph->getNode(decNode->eEdge1.Node);

    // get the AIG nodes corresponding to the children
    lhsAnd = thData->decNodeFunc[lhsNode->id];
    rhsAnd = thData->decNodeFunc[rhsNode->id];

		if ( lhsAnd && rhsAnd ) {
			if (lhsNode->id < 4) { // If lhs is a cut leaf
				lhsAndPol = decNode->eEdge0.fCompl
												? !(thData->currentFaninsPol[lhsNode->id])
												: thData->currentFaninsPol[lhsNode->id];
			} else {
				lhsAndPol = decNode->eEdge0.fCompl ? false : true;
			}

			if (rhsNode->id < 4) { // If rhs is a cut leaf
				rhsAndPol = decNode->eEdge1.fCompl
												? !(thData->currentFaninsPol[rhsNode->id])
												: thData->currentFaninsPol[rhsNode->id];
			} else {
				rhsAndPol = decNode->eEdge1.fCompl ? false : true;
			}

			curAnd = this->aig.lookupNodeInFanoutMap(lhsAnd, rhsAnd, lhsAndPol, rhsAndPol);

			if (curAnd) {
				aig::NodeData& curAndData = aigGraph.getData( curAnd, galois::MethodFlag::READ );
				if ( curAndData.nFanout == 0 ) {
					return false;
				}
			}
		}
		else {
			curAnd = nullptr;
		}

    thData->decNodeFunc[decNode->id] = curAnd;
  }

  aig::GNode choiceRoot = thData->decNodeFunc[decNode->id];

	if ( choiceRoot != nullptr ) {
		aig::NodeData& choiceRootData = aigGraph.getData( choiceRoot, galois::MethodFlag::READ );
		if ( choiceRootData.nFanout > 0 ) {
			return false;
		}
	}

  return true;
}

/* Transforms the decomposition graph into the AIG. Before calling this procedure, AIG nodes 
 * for the fanins (cut's leaves) should be assigned to thData->decNodeFun[ decNode.id ]. */
aig::GNode ChoiceManager::decGraphToAigCreate(ThreadLocalDataCH* thData, DecGraph* decGraph) {

  DecNode* decNode;
  DecNode* lhsNode;
  DecNode* rhsNode;
  aig::GNode curAnd;
  aig::GNode lhsAnd;
  aig::GNode rhsAnd;
  bool lhsAndPol;
  bool rhsAndPol;
	aig::Graph& aigGraph = this->aig.getGraph();

  // build the AIG nodes corresponding to the AND gates of the graph
  for (int i = decGraph->getLeaveNum(); (i < decGraph->getNodeNum()) && ((decNode = decGraph->getNode(i)), 1); i++) {

    // get the children of this node
    lhsNode = decGraph->getNode(decNode->eEdge0.Node);
    rhsNode = decGraph->getNode(decNode->eEdge1.Node);

    // get the AIG nodes corresponding to the children
    lhsAnd = thData->decNodeFunc[lhsNode->id];
    rhsAnd = thData->decNodeFunc[rhsNode->id];

    if (lhsNode->id < 4) { // If lhs is a cut leaf
      lhsAndPol = decNode->eEdge0.fCompl
                      ? !(thData->currentFaninsPol[lhsNode->id])
                      : thData->currentFaninsPol[lhsNode->id];
    } else {
      lhsAndPol = decNode->eEdge0.fCompl ? false : true;
    }

    if (rhsNode->id < 4) { // If rhs is a cut leaf
      rhsAndPol = decNode->eEdge1.fCompl
                      ? !(thData->currentFaninsPol[rhsNode->id])
                      : thData->currentFaninsPol[rhsNode->id];
    } else {
      rhsAndPol = decNode->eEdge1.fCompl ? false : true;
    }

    curAnd = this->aig.lookupNodeInFanoutMap(lhsAnd, rhsAnd, lhsAndPol, rhsAndPol);

    if (curAnd) {
      thData->decNodeFunc[decNode->id] = curAnd;
    } else {
      thData->decNodeFunc[decNode->id] = this->aig.createAND(lhsAnd, rhsAnd, lhsAndPol, rhsAndPol);
			aig::NodeData& newNodeData = aigGraph.getData( thData->decNodeFunc[decNode->id], galois::MethodFlag::WRITE );
			newNodeData.counter = 3; // Mark as processed to avoind to insert it into the worklist.
    }
  }

  return thData->decNodeFunc[decNode->id];
}

void ChoiceManager::lockFaninCone(aig::Graph& aigGraph, aig::GNode node, Cut* cut) {

  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ); // lock

  // If node is a cut leaf
  if ((nodeData.id == cut->leaves[0]) || (nodeData.id == cut->leaves[1]) ||
      (nodeData.id == cut->leaves[2]) || (nodeData.id == cut->leaves[3])) {
    return;
  }

  // If node is a PI
  if ( (nodeData.type == aig::NodeType::PI) || (nodeData.type == aig::NodeType::LATCH) ) {
    return;
  }

  auto inEdgeIt      = aigGraph.in_edge_begin(node);
  aig::GNode lhsNode = aigGraph.getEdgeDst(inEdgeIt);
  aig::NodeData& lhsData = aigGraph.getData(lhsNode, galois::MethodFlag::READ); // lock
  inEdgeIt++;
  aig::GNode rhsNode = aigGraph.getEdgeDst(inEdgeIt);
  aig::NodeData& rhsData = aigGraph.getData(rhsNode, galois::MethodFlag::READ); // lock

  lockFaninCone(aigGraph, lhsNode, cut);
  lockFaninCone(aigGraph, rhsNode, cut);
}

aig::Aig& ChoiceManager::getAig() { return this->aig; }

CutManager& ChoiceManager::getCutMan() { return this->cutMan; }

NPNManager& ChoiceManager::getNPNMan() { return this->npnMan; }

PreCompGraphManager& ChoiceManager::getPcgMan() { return this->pcgMan; }

PerThreadDataCH& ChoiceManager::getPerThreadDataCH() {
  return this->perThreadDataCH;
}

struct ChoiceOperator {

  ChoiceManager& chMan;
  CutManager& cutMan;

  ChoiceOperator(ChoiceManager& chMan) : chMan(chMan), cutMan(chMan.getCutMan()) {}

  void operator()(aig::GNode node, galois::UserContext<aig::GNode>& ctx) {

    aig::Graph& aigGraph = chMan.getAig().getGraph();

    if ((node == nullptr) || !aigGraph.containsNode(node, galois::MethodFlag::WRITE)) {
      return;
    }

    aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::WRITE);

		if ( nodeData.counter == 3 ) {
			return;
		}

	  // Touching outgoing neighobors to acquire their locks
    for (auto outEdge : aigGraph.out_edges(node)) {}

    if (nodeData.type == aig::NodeType::AND) {
			ThreadLocalDataCH* thData = chMan.getPerThreadDataCH().getLocal();
			chMan.createNodeChoices(thData, node);
			/*
			aig::GNode currChoice = nodeData.choiceList;
			while ( currChoice != nullptr ) {
				aig::NodeData& currChoiceData = aigGraph.getData( currChoice, galois::MethodFlag::READ );
				std::cout << "Node " << nodeData.id << " -> Choice Node " << currChoiceData.id << std::endl;
				currChoice = currChoiceData.choiceList;
			}	
			*/
    } 
		else {
      if ( (nodeData.type == aig::NodeType::PI) || (nodeData.type == aig::NodeType::LATCH) ) {
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
      }
    }

		// Schedule fanout nodes
	 	if (nodeData.counter == 2) {
			nodeData.counter += 1;
		}
		if (nodeData.counter == 3) {
			// Insert nextNodes in the worklist
			for (auto outEdge : aigGraph.out_edges(node)) {
				aig::GNode nextNode = aigGraph.getEdgeDst(outEdge);
				aig::NodeData& nextNodeData = aigGraph.getData(nextNode, galois::MethodFlag::WRITE);

				if ( ( nextNodeData.type == aig::NodeType::PO ) || ( nextNodeData.type == aig::NodeType::LATCH ) ) {
					continue;
				}

				nextNodeData.counter += 1;
				if (nextNodeData.counter == 2) {
					if (cutMan.getNodeCuts()[nextNodeData.id] != nullptr) {
						cutMan.recycleNodeCuts(nextNodeData.id);
					}
					ctx.push(nextNode);
				}
			}
		}

  }
};

void runChoiceOperator(ChoiceManager& chMan) {

  galois::InsertBag<aig::GNode> workList;
  typedef galois::worklists::PerSocketChunkBag<500> DC_BAG;

  for (auto pi : chMan.getAig().getInputNodes()) {
    workList.push(pi);
  }

  //for (auto latch : chMan.getAig().getLatchNodes()) {
  //  workList.push(latch);
  //}

  // Galois Parallel Foreach
  galois::for_each(galois::iterate(workList.begin(), workList.end()),
                   ChoiceOperator(chMan), 
									 galois::wl<DC_BAG>(),
                   galois::loopname("ChoiceOperator"),
                   galois::per_iter_alloc());

}

} /* namespace algorithm */
