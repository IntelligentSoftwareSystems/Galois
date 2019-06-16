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
 Parallel Parallel LUT-Based Tech Mapping October 16, 2018.
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#include "PriorityCutManager.h"
#include "galois/Galois.h"
#include "galois/Bag.h"

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <assert.h>

using namespace std::chrono;

namespace algorithm {

PriCutManager::PriCutManager(aig::Aig& aig, int K, int C, int nThreads, bool compTruth) 
			:
      aig(aig), K(K), C(C), nThreads(nThreads), compTruth(compTruth),
      nWords(Functional32::wordNum(K)),
      nNodes(std::distance(aig.getGraph().begin(), aig.getGraph().end()) - aig.getNumOutputs()),
      cutPoolSize(nNodes / nThreads),
      perThreadPriCutPool(cutPoolSize, K, compTruth), perThreadPriCutList(C), perThreadAuxTruth(nWords) {

	nLUTs = 0;
	nLevels = 0;
	sortMode = 1;
	fPower = false;
	fEpsilon = (float)0.005;

  kcutTime = 0;
  nodePriCuts = new PriCut*[nNodes + 1];
  for (int i = 0; i < nNodes + 1; i++) {
    nodePriCuts[i] = nullptr;
  }
}

PriCutManager::~PriCutManager() { delete[] nodePriCuts; }

void PriCutManager::computePriCutsRecursively(aig::GNode node) {

  aig::Graph& aigGraph    = this->aig.getGraph();
  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

  if (this->nodePriCuts[nodeData.id] == nullptr) {

    auto inEdgeIt          = aigGraph.in_edge_begin(node);
    aig::GNode lhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& lhsData = aigGraph.getData(lhsNode);
    bool lhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    inEdgeIt++;
    aig::GNode rhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& rhsData = aigGraph.getData(rhsNode);
    bool rhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    PriCutPool* cutPool    = this->perThreadPriCutPool.getLocal();
    PriCutList* cutList    = this->perThreadPriCutList.getLocal();
    AuxTruth* auxTruth 		 = this->perThreadAuxTruth.getLocal();

    computePriCutsRec(lhsNode, cutPool, cutList, auxTruth);
    computePriCutsRec(rhsNode, cutPool, cutList, auxTruth);

    computePriCuts(cutPool, cutList, auxTruth, nodeData.id, lhsData.id, rhsData.id,
                lhsPolarity, rhsPolarity);
  }
}

void PriCutManager::computePriCutsRec(aig::GNode node, PriCutPool* cutPool,
                                PriCutList* cutList, AuxTruth* auxTruth) {

  aig::Graph& aigGraph    = this->aig.getGraph();
  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

  if (this->nodePriCuts[nodeData.id] == nullptr) {

    auto inEdgeIt          = aigGraph.in_edge_begin(node);
    aig::GNode lhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& lhsData = aigGraph.getData(lhsNode);
    bool lhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    inEdgeIt++;
    aig::GNode rhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& rhsData = aigGraph.getData(rhsNode);
    bool rhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    computePriCutsRec(lhsNode, cutPool, cutList, auxTruth);
    computePriCutsRec(rhsNode, cutPool, cutList, auxTruth);

    computePriCuts(cutPool, cutList, auxTruth, nodeData.id, lhsData.id, rhsData.id,
                lhsPolarity, rhsPolarity);
  }
}

void PriCutManager::computePriCuts(PriCutPool* cutPool, PriCutList* cutList,
                             AuxTruth* auxTruth, int nodeId, int lhsId,
                             int rhsId, bool lhsPolarity, bool rhsPolarity) {

  int i;
	PriCut * trivialCut, * resCut;

  for (PriCut* lhsCut = this->nodePriCuts[lhsId]; lhsCut != nullptr; lhsCut = lhsCut->nextCut) {
    for (PriCut* rhsCut = this->nodePriCuts[rhsId]; rhsCut != nullptr; rhsCut = rhsCut->nextCut) {
			
			if (Functional32::countOnes( lhsCut->sig | rhsCut->sig ) > this->K) {
				continue;
			}

  		// merge the cuts
  		if (lhsCut->nLeaves >= rhsCut->nLeaves) {
   			resCut = mergeCuts(cutPool, lhsCut, rhsCut);
  		} else {
    		resCut = mergeCuts(cutPool, rhsCut, lhsCut);
  		}

  		if (resCut == nullptr) {
				continue;
  		}

  		// check containment
  		if (cutFilter(cutPool, cutList, resCut)) {
  			continue;
			}

  		if (this->compTruth) {
    		computeTruth(auxTruth, resCut, lhsCut, rhsCut, lhsPolarity, rhsPolarity);
				//std::cout << Functional32::toHex( readTruth( resCut ), getNWords() ) << std::endl;
  		}

			if ( this->sortMode == 2 ) {
				resCut->area = cutAreaDerefed( resCut ); // FIXME to use this heuristic the locks of neighbors must to be aquired first
			}
			else {
				resCut->area = cutAreaFlow( resCut );
			}

			resCut->delay = cutDelay( resCut );
			resCut->edge = cutEdgeFlow( resCut );

  		// add to the sorted list
  		cutSort(cutPool, cutList, resCut);
  	}
  }

 	// start with the elementary cut
  trivialCut = cutPool->getMemory();
  trivialCut->leaves[0] = nodeId;
  trivialCut->nLeaves++;
  trivialCut->sig = (1U << (nodeId % 31));
  if (this->compTruth) {
    unsigned* cutTruth = readTruth(trivialCut);
    for (int i = 0; i < this->nWords; i++) {
      cutTruth[i] = 0xAAAAAAAA;
    }
  }
  cutList->array[cutList->nCuts++] = trivialCut;
  nCuts += 1;
  nTriv += 1;

  // Copy from currentCutList to the nodeCuts
  commitCuts(nodeId, cutList);
}

PriCut* PriCutManager::mergeCuts(PriCutPool* cutPool, PriCut* lhsCut, PriCut* rhsCut) {

  // assert( lhsCut->nLeaves >= rhsCut->nLeaves );
  int i, j, l;
  PriCut* resCut;

  // the case of the largest cut sizes
  if (lhsCut->nLeaves == this->K && rhsCut->nLeaves == this->K) {
    for (i = 0; i < lhsCut->nLeaves; i++) {
      if (lhsCut->leaves[i] != rhsCut->leaves[i]) {
        return nullptr;
      }
    }
    resCut = cutPool->getMemory();
    for (i = 0; i < lhsCut->nLeaves; i++) {
      resCut->leaves[i] = lhsCut->leaves[i];
    }
    resCut->nLeaves = lhsCut->nLeaves;
  	resCut->sig = lhsCut->sig | rhsCut->sig; // set the signature
    return resCut;
  }

  // the case when one of the cuts is the largest
  if (lhsCut->nLeaves == this->K) {
    for (i = 0; i < rhsCut->nLeaves; i++) {
      for (j = lhsCut->nLeaves - 1; j >= 0; j--) {
        if (lhsCut->leaves[j] == rhsCut->leaves[i]) {
          break;
        }
      }
      if (j == -1) { // did not find
        return nullptr;
      }
    }
    resCut = cutPool->getMemory();
    for (i = 0; i < lhsCut->nLeaves; i++) {
      resCut->leaves[i] = lhsCut->leaves[i];
    }
    resCut->nLeaves = lhsCut->nLeaves;
  	resCut->sig = lhsCut->sig | rhsCut->sig; // set the signature
    return resCut;
  }

  // compare two cuts with different numbers
  resCut = cutPool->getMemory();
  i = 0;
  j = 0;
  for (l = 0; l < this->K; l++) {
    if (j == rhsCut->nLeaves) {
      if (i == lhsCut->nLeaves) {
        resCut->nLeaves = l;
  			resCut->sig = lhsCut->sig | rhsCut->sig; // set the signature
        return resCut;
      }
      resCut->leaves[l] = lhsCut->leaves[i++];
      continue;
    }

    if (i == lhsCut->nLeaves) {
      if (j == rhsCut->nLeaves) {
        resCut->nLeaves = l;
  			resCut->sig = lhsCut->sig | rhsCut->sig; // set the signature
        return resCut;
      }
      resCut->leaves[l] = rhsCut->leaves[j++];
      continue;
    }

    if (lhsCut->leaves[i] < rhsCut->leaves[j]) {
      resCut->leaves[l] = lhsCut->leaves[i++];
      continue;
    }

    if (lhsCut->leaves[i] > rhsCut->leaves[j]) {
      resCut->leaves[l] = rhsCut->leaves[j++];
      continue;
    }

    resCut->leaves[l] = lhsCut->leaves[i++];
    j++;
  }

  if (i < lhsCut->nLeaves || j < rhsCut->nLeaves) {
    cutPool->giveBackMemory(resCut);
    return nullptr;
  }

  resCut->nLeaves = l;
  resCut->sig = lhsCut->sig | rhsCut->sig; // set the signature
  return resCut;
}

inline bool PriCutManager::cutFilter(PriCutPool* cutPool, PriCutList* cutList, PriCut* resCut) {

	PriCut * cut;

	for (int i = 0; i < cutList->nCuts; i++ ) {

		cut = cutList->array[i];

		if (cut->nLeaves <= resCut->nLeaves) {
	    // skip the non-contained cuts
   		if ((cut->sig & resCut->sig) != cut->sig) {	
    		continue;
    	}
    	// check containment seriously
    	if (checkCutDominance(cut, resCut)) {
      	cutPool->giveBackMemory(resCut); // Recycle Cut
      	nFilt += 1;
    		return true; // resCut is dominated
  		}
		} else {
			 // sKip the non-contained cuts
      if ((cut->sig & resCut->sig) != resCut->sig) {
      	continue;
      }
      // check containment seriously
      if (checkCutDominance(resCut, cut)) {
        nCuts -= 1;
        nFilt += 1; 
				cutList->nCuts--;
				cutPool->giveBackMemory(cut); // Recycle Cut
				for (int j = i; j < cutList->nCuts; j++) {
					cutList->array[j] = cutList->array[j+1]; 
				}
      }
		}	
  }
  return false;
}

inline bool PriCutManager::checkCutDominance(PriCut* smallerCut, PriCut* largerCut) {
	
	int i, j;
  for (i = 0; i < smallerCut->nLeaves; i++) {
    for (j = 0; j < largerCut->nLeaves; j++) {
      if (smallerCut->leaves[i] == largerCut->leaves[j]) {
        break;
      }
    }
    if (j == largerCut->nLeaves) { // node i in smallerCut is not contained in largerCut
      return false;
    }
  }
  // every node in smallerCut is contained in largerCut
  return true;
}

void PriCutManager::commitCuts(int nodeId, PriCutList* cutList) {

	assert(cutList->nCuts != 0);

  // Copy from currenti CutList to the nodePriCuts and clean up the cutList
  this->nodePriCuts[nodeId] = cutList->array[0];

  int i;
	for ( i = 0; i < cutList->nCuts-1; i++) {
		cutList->array[i]->nextCut = cutList->array[i+1];
		cutList->array[i] = nullptr;
	}
	cutList->array[i]->nextCut = nullptr;
	cutList->array[i] = nullptr;
	cutList->nCuts = 0;
}

void PriCutManager::computeTruth(AuxTruth* auxTruth, PriCut* resCut, PriCut* lhsCut,
                              PriCut* rhsCut, bool lhsPolarity, bool rhsPolarity) {

  // permute the first table
  if (lhsPolarity) {
    Functional32::copy(auxTruth->truth[0], readTruth(lhsCut), this->nWords);
  } else {
    Functional32::NOT(auxTruth->truth[0], readTruth(lhsCut), this->nWords);
  }
  Functional32::truthStretch(auxTruth->truth[2], auxTruth->truth[0],
                             lhsCut->nLeaves, this->K,
                             truthPhase(resCut, lhsCut));

  // permute the second table
  if (rhsPolarity) {
    Functional32::copy(auxTruth->truth[1], readTruth(rhsCut), this->nWords);
  } else {
    Functional32::NOT(auxTruth->truth[1], readTruth(rhsCut), this->nWords);
  }
  Functional32::truthStretch(auxTruth->truth[3], auxTruth->truth[1],
                             rhsCut->nLeaves, this->K,
                             truthPhase(resCut, rhsCut));

  // produce the resulting table. In this first version we are not considering
  // the cut->fCompl flag. It may be considerer in further versions according to
  // the demand.
  // if ( cut->fCompl ) {
  //	Functional32::NAND( readTruth( cut ) , auxTruth[2], auxTruth[3], K );
  //}
  // else {
  Functional32::AND(readTruth(resCut), auxTruth->truth[2], auxTruth->truth[3],
                    this->nWords);
  //}
}

inline unsigned PriCutManager::truthPhase(PriCut* resCut, PriCut* inCut) {

  unsigned phase = 0;
  int i, j;
  for (i = j = 0; i < resCut->nLeaves; i++) {

    if (j == inCut->nLeaves) {
      break;
    }
    if (resCut->leaves[i] < inCut->leaves[j]) {
      continue;
    }

    assert(resCut->leaves[i] == inCut->leaves[j]);
    phase |= (1 << i);
    j++;
  }

  return phase;
}

unsigned int* PriCutManager::readTruth(PriCut* cut) {
  return (unsigned*)(cut->leaves + this->K);
}

/*
 *     This method gives the cut's memory back to current thread cutPool.
 *     However, the memory can be allocated by the cutPool of one thread
 *     and returned to cutPool of another thread.
 */
void PriCutManager::recycleNodeCuts(int nodeId) {

  PriCutPool* cutPool = this->perThreadPriCutPool.getLocal();

  for (PriCut* cut = this->nodePriCuts[nodeId]; cut != nullptr; cut = cut->nextCut) {
    cutPool->giveBackMemory(cut);
  }

  this->nodePriCuts[nodeId] = nullptr;
}

void PriCutManager::cutSort(PriCutPool* cutPool, PriCutList * cutList, PriCut * resCut) {
 
	// cut structure is empty
	if (cutList->nCuts == 0) {
		cutList->array[cutList->nCuts++] = resCut;
  	nCuts += 1;
		return;
	}

	// the cut will be added - find its place
	cutList->array[cutList->nCuts++] = resCut;

	for (int i = cutList->nCuts-2; i >= 0 ; i--) {
		if (sortCompare( cutList->array[i], resCut ) <= 0)
			break;
		cutList->array[i+1] = cutList->array[i];
		cutList->array[i] = resCut;
	}
	
	if (cutList->nCuts > this->C) {
		cutPool->giveBackMemory(cutList->array[--cutList->nCuts]);
	}
	else {
  	nCuts += 1;
	}
}

int PriCutManager::sortCompare(PriCut * lhsCut, PriCut * rhsCut) {

    if ( this->fPower ) {
        if ( this->sortMode == 1 ) { // area flow       
            if ( lhsCut->area < rhsCut->area - this->fEpsilon )
                return -1;
            if ( lhsCut->area > rhsCut->area + this->fEpsilon )
                return 1;
            if ( lhsCut->power < rhsCut->power - this->fEpsilon )
                return -1;
            if ( lhsCut->power > rhsCut->power + this->fEpsilon )
                return 1;
            if ( lhsCut->edge < rhsCut->edge - this->fEpsilon )
                return -1;
            if ( lhsCut->edge > rhsCut->edge + this->fEpsilon )
                return 1;
            if ( lhsCut->nLeaves < rhsCut->nLeaves )
                return -1;
            if ( lhsCut->nLeaves > rhsCut->nLeaves )
                return 1;
            if ( lhsCut->delay < rhsCut->delay - this->fEpsilon )
                return -1;
            if ( lhsCut->delay > rhsCut->delay + this->fEpsilon )
                return 1;
            return 0;
        }
        if ( this->sortMode == 0 ) { // delay
            if ( lhsCut->delay < rhsCut->delay - this->fEpsilon )
                return -1;
            if ( lhsCut->delay > rhsCut->delay + this->fEpsilon )
                return 1;
            if ( lhsCut->nLeaves < rhsCut->nLeaves )
                return -1;
            if ( lhsCut->nLeaves > rhsCut->nLeaves )
                return 1;
            if ( lhsCut->area < rhsCut->area - this->fEpsilon )
                return -1;
            if ( lhsCut->area > rhsCut->area + this->fEpsilon )
                return 1;
            if ( lhsCut->power < rhsCut->power - this->fEpsilon  )
                return -1;
            if ( lhsCut->power > rhsCut->power + this->fEpsilon  )
                return 1;
            if ( lhsCut->edge < rhsCut->edge - this->fEpsilon )
                return -1;
            if ( lhsCut->edge > rhsCut->edge + this->fEpsilon )
                return 1;
            return 0;
        }
        assert( this->sortMode == 2 ); // delay old, exact area
        if ( lhsCut->delay < rhsCut->delay - this->fEpsilon )
            return -1;
        if ( lhsCut->delay > rhsCut->delay + this->fEpsilon )
            return 1;
        if ( lhsCut->power < rhsCut->power - this->fEpsilon  )
            return -1;
        if ( lhsCut->power > rhsCut->power + this->fEpsilon  )
            return 1;
        if ( lhsCut->edge < rhsCut->edge - this->fEpsilon )
            return -1;
        if ( lhsCut->edge > rhsCut->edge + this->fEpsilon )
            return 1;
        if ( lhsCut->area < rhsCut->area - this->fEpsilon )
            return -1;
        if ( lhsCut->area > rhsCut->area + this->fEpsilon )
            return 1;
        if ( lhsCut->nLeaves < rhsCut->nLeaves )
            return -1;
        if ( lhsCut->nLeaves > rhsCut->nLeaves )
            return 1;
        return 0;
    } 
    else  { // regular
        if ( this->sortMode == 1 ) { // area
            if ( lhsCut->area < rhsCut->area - this->fEpsilon )
                return -1;
            if ( lhsCut->area > rhsCut->area + this->fEpsilon )
                return 1;
            if ( lhsCut->edge < rhsCut->edge - this->fEpsilon )
                return -1;
            if ( lhsCut->edge > rhsCut->edge + this->fEpsilon )
                return 1;
            if ( lhsCut->power < rhsCut->power - this->fEpsilon )
                return -1;
            if ( lhsCut->power > rhsCut->power + this->fEpsilon )
                return 1;
            if ( lhsCut->nLeaves < rhsCut->nLeaves )
                return -1;
            if ( lhsCut->nLeaves > rhsCut->nLeaves )
                return 1;
            if ( lhsCut->delay < rhsCut->delay - this->fEpsilon )
                return -1;
            if ( lhsCut->delay > rhsCut->delay + this->fEpsilon )
                return 1;
            return 0;
        }
        if ( this->sortMode == 0 ) { // delay
            if ( lhsCut->delay < rhsCut->delay - this->fEpsilon )
                return -1;
            if ( lhsCut->delay > rhsCut->delay + this->fEpsilon )
                return 1;
            if ( lhsCut->nLeaves < rhsCut->nLeaves )
                return -1;
            if ( lhsCut->nLeaves > rhsCut->nLeaves )
                return 1;
            if ( lhsCut->area < rhsCut->area - this->fEpsilon )
                return -1;
            if ( lhsCut->area > rhsCut->area + this->fEpsilon )
                return 1;
            if ( lhsCut->edge < rhsCut->edge - this->fEpsilon )
                return -1;
            if ( lhsCut->edge > rhsCut->edge + this->fEpsilon )
                return 1;
            if ( lhsCut->power < rhsCut->power - this->fEpsilon )
                return -1;
            if ( lhsCut->power > rhsCut->power + this->fEpsilon )
                return 1;
            return 0;
        }
        assert( this->sortMode == 2 ); // delay old
        if ( lhsCut->delay < rhsCut->delay - this->fEpsilon )
            return -1;
        if ( lhsCut->delay > rhsCut->delay + this->fEpsilon )
            return 1;
        if ( lhsCut->area < rhsCut->area - this->fEpsilon )
            return -1;
        if ( lhsCut->area > rhsCut->area + this->fEpsilon )
            return 1;
        if ( lhsCut->edge < rhsCut->edge - this->fEpsilon )
            return -1;
        if ( lhsCut->edge > rhsCut->edge + this->fEpsilon )
            return 1;
        if ( lhsCut->power < rhsCut->power - this->fEpsilon )
            return -1;
        if ( lhsCut->power > rhsCut->power + this->fEpsilon )
            return 1;
        if ( lhsCut->nLeaves < rhsCut->nLeaves )
            return -1;
        if ( lhsCut->nLeaves > rhsCut->nLeaves )
            return 1;
        return 0;
    }
}

float PriCutManager::cutAreaFlow( PriCut * cut ) {

  aig::Graph & aigGraph = this->aig.getGraph();
	int leafId;
	float flow = 1.0;

	for ( int i = 0 ; i < cut->nLeaves ; i++ ) {

		leafId = cut->leaves[i];
		aig::GNode leaf = this->aig.getNodes()[ leafId ];
		aig::NodeData & leafData = aigGraph.getData( leaf, galois::MethodFlag::UNPROTECTED ); // Note: if the graph topology is not changed, it dont need to lock leaves.

		if ( ( leafData.nFanout == 0 ) || ( leafData.type == aig::NodeType::CONSTZERO ) ) {
			flow += getBestCut( leafId )->area; 
		}
		else 
		{
			assert( leafData.nFanout != 0 );
			flow += getBestCut( leafId )->area / leafData.nFanout;
		}
  }
	return flow;
}

float PriCutManager::cutAreaDerefed( PriCut * cut ) {

	float aResult, aResult2;

	if ( cut->nLeaves < 2 )
		return 0;
	aResult2 = cutAreaRef( cut );
	aResult  = cutAreaDeref( cut );
	assert( aResult > aResult2 - this->fEpsilon );
	assert( aResult < aResult2 + this->fEpsilon );
	return aResult;
}

float PriCutManager::cutAreaRef( PriCut * cut ) {

  aig::Graph & aigGraph = this->aig.getGraph();
	int leafId;
	float area = 1.0;	

	for ( int i = 0 ; i < cut->nLeaves ; i++ ) {

		leafId = cut->leaves[i];
		aig::GNode leaf = this->aig.getNodes()[ leafId ];
		aig::NodeData & leafData = aigGraph.getData( leaf, galois::MethodFlag::READ ); // FIXME lock neigborhood earlier

		assert( leafData.nFanout >= 0 );

		if ( ( leafData.nFanout++ > 0 ) || ( leafData.type != aig::NodeType::AND ) )
			continue;

		area += cutAreaRef( getBestCut( leafId ) );
	}
	return area;
}

float PriCutManager::cutAreaDeref( PriCut * cut ) {

  aig::Graph & aigGraph = this->aig.getGraph();
	int leafId;
	float area = 1.0;	

	for ( int i = 0 ; i < cut->nLeaves ; i++ ) {

		leafId = cut->leaves[i];
		aig::GNode leaf = this->aig.getNodes()[ leafId ];
		aig::NodeData & leafData = aigGraph.getData( leaf, galois::MethodFlag::READ ); // FIXME lock neigborhood earlier

		assert( leafData.nFanout > 0 );

		if ( --leafData.nFanout > 0 || ( leafData.type != aig::NodeType::AND ) )
			continue;

		area += cutAreaDeref( getBestCut( leafId ) );
	}
	return area;
}

float PriCutManager::cutDelay( PriCut * cut ) {

	int leafId;
	float currDelay, delay = std::numeric_limits<float>::min();

	for ( int i = 0 ; i < cut->nLeaves ; i++ ) {
		leafId = cut->leaves[i];
		currDelay = getBestCut( leafId )->delay + 1.0;
		delay = std::max( delay, currDelay );
	}
	return delay;
}

float PriCutManager::cutEdgeFlow( PriCut * cut ) {

	aig::Graph & aigGraph = this->aig.getGraph();
	int leafId;
	float flow = cut->nLeaves;

	for ( int i = 0 ; i < cut->nLeaves ; i++ ) {

		leafId = cut->leaves[i];
		aig::GNode leaf = this->aig.getNodes()[ leafId ];
		aig::NodeData & leafData = aigGraph.getData( leaf, galois::MethodFlag::UNPROTECTED ); // Note: if the graph topology is not changed, it dont need to lock leaves.

    if ( ( leafData.nFanout == 0 ) || ( leafData.type == aig::NodeType::CONSTZERO ) ) {
			flow += getBestCut( leafId )->edge;
		}
		else 
		{
			assert( leafData.nFanout > this->fEpsilon );
			flow += getBestCut( leafId )->edge / leafData.nFanout;
		}
	}
  return flow;
}


PriCut * PriCutManager::getBestCut( int nodeId ) {
	return this->nodePriCuts[ nodeId ]; // the first cut is the best cut
}


void PriCutManager::computeCovering() { // FIXME to consider LATCHES

  aig::Graph& aigGraph = this->aig.getGraph();
	PriCut * bestCut;
	aig::GNode leaf;
	int currLevel, maxLevel;

  for (auto po : this->aig.getOutputNodes()) {	
    auto inEdgeIt = aigGraph.in_edge_begin( po );
		aig::GNode inNode = aigGraph.getEdgeDst( inEdgeIt );
		aig::NodeData& inNodeData = aigGraph.getData( inNode, galois::MethodFlag::UNPROTECTED ); // It will be executed serially
		maxLevel = 0;

		if ( ( inNodeData.type == aig::NodeType::PI ) 		||
				 ( inNodeData.type == aig::NodeType::LATCH ) 	||
				 ( inNodeData.type == aig::NodeType::CONSTZERO ) ) {
			continue;
		}

		auto it = this->covering.find( inNodeData.id );

		if ( it == this->covering.end() ) {

			bestCut = getBestCut( inNodeData.id );
			//printNodeBestCut( inNodeData.id );

			for ( int i = 0; i < bestCut->nLeaves; i++ ) {
				leaf = this->aig.getNodes()[ bestCut->leaves[i] ];
				currLevel = computeCoveringRec( aigGraph, leaf );
				if ( maxLevel < currLevel ) {
					maxLevel = currLevel;
				}
			}

			LUT newLUT;
			newLUT.bestCut = bestCut;
			newLUT.rootId = inNodeData.id;
			newLUT.level = 1 + maxLevel;
			maxLevel = newLUT.level;
			this->covering.insert( { inNodeData.id, newLUT } );
		}
		else {
			maxLevel = it->second.level;
		}

		if ( this->nLevels < maxLevel ) {
			this->nLevels = maxLevel;
		}
	}
}

int PriCutManager::computeCoveringRec( aig::Graph & aigGraph, aig::GNode node ) { // FIXME to consider LATCHES

	PriCut * bestCut;
	aig::GNode leaf;
	int currLevel, maxLevel = 0;

	aig::NodeData& nodeData = aigGraph.getData( node, galois::MethodFlag::READ );
	
	if ( ( nodeData.type == aig::NodeType::PI ) 		|| 
			 ( nodeData.type == aig::NodeType::LATCH ) 	||
			 ( nodeData.type == aig::NodeType::CONSTZERO ) ) {
		return 0;
	}

	auto it = this->covering.find( nodeData.id );

	if ( it == this->covering.end() ) {

		bestCut = getBestCut( nodeData.id );
		//printNodeBestCut( nodeData.id );

		for ( int i = 0; i < bestCut->nLeaves; i++ ) {
			leaf = this->aig.getNodes()[ bestCut->leaves[i] ];
			currLevel = computeCoveringRec( aigGraph, leaf );
			if ( maxLevel < currLevel ) {
					maxLevel = currLevel;
			}
		}
		
		LUT newLUT;
		newLUT.bestCut = bestCut;
		newLUT.rootId = nodeData.id;
		newLUT.level = 1 + maxLevel;
		this->covering.insert( { nodeData.id, newLUT } );
		return newLUT.level;
	}
	else {
		return it->second.level;
	}
}

int PriCutManager::getNumLUTs() {
	this->nLUTs = this->covering.size();
	return this->nLUTs;
}

int PriCutManager::getNumLevels() {
	return this->nLevels;
}

void PriCutManager::printCovering() {

  std::cout << std::endl << "########## Mapping Covering ###############" << std::endl;
	PriCut * bestCut;
	for ( auto entry : this->covering ) {
		std::cout << "Node " << entry.first << ": { ";
		bestCut = entry.second.bestCut;		
	  for (int i = 0; i < bestCut->nLeaves; i++) {
  	  std::cout << bestCut->leaves[i] << " ";
	  }
  	std::cout << "}" << std::endl;
	  //std::cout << "}[" << Functional32::toHex( readTruth( bestCut ), this->nWords )  << "] " << std::endl;
	}
  std::cout << std::endl << "###########################################" << std::endl;
}

void PriCutManager::printNodeCuts(int nodeId, long int& counter) {

  std::cout << "Node " << nodeId << ": { ";
  for (PriCut* currentCut = this->nodePriCuts[nodeId]; currentCut != nullptr;
       currentCut      = currentCut->nextCut) {
    counter++;
    std::cout << "{ ";
    for (int i = 0; i < currentCut->nLeaves; i++) {
      std::cout << currentCut->leaves[i] << " ";
    }
    std::cout << "}(" << currentCut->area << ") ";
    //std::cout << "}[" << Functional32::toHex( readTruth( currentCut ), this->nWords )  << "] ";
  }
  std::cout << "}" << std::endl;
}

void PriCutManager::printAllCuts() {

  long int counter     = 0;
  aig::Graph& aigGraph = this->aig.getGraph();

  std::cout << std::endl << "########## All K-Cuts ###########" << std::endl;
  for (aig::GNode node : aigGraph) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);
    if ((nodeData.type == aig::NodeType::AND) ||
        (nodeData.type == aig::NodeType::PI)) {
      printNodeCuts(nodeData.id, counter);
    }
  }
  std::cout << "#################################" << std::endl;
}

void PriCutManager::printNodeBestCut(int nodeId) {

	PriCut * bestCut = getBestCut( nodeId );
  std::cout << "Node " << nodeId << ": { ";
  for (int i = 0; i < bestCut->nLeaves; i++) {
    std::cout << bestCut->leaves[i] << " ";
  }
  std::cout << "}(" << bestCut->area << ")" << std::endl;
  //std::cout << "}[" << Functional32::toHex( readTruth( bestCut ), this->nWords )  << "] " << std::endl;
}

void PriCutManager::printBestCuts() {

  aig::Graph& aigGraph = this->aig.getGraph();

  std::cout << std::endl << "########## Best K-Cuts ###########" << std::endl;
  for (aig::GNode node : aigGraph) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);
    if ((nodeData.type == aig::NodeType::AND) ||
        (nodeData.type == aig::NodeType::PI)) {
				printNodeBestCut( nodeData.id );
    }
  }
  std::cout << "#################################" << std::endl;
}

void PriCutManager::printCutStatistics() {

  long int nCutsRed = nCuts.reduce();
  nCutsRed += this->aig.getNumInputs();

  long int nTrivRed = nTriv.reduce();
  nTrivRed += this->aig.getNumInputs();

  long int nFiltRed = nFilt.reduce();

  long int nSatuRed = nSatu.reduce();


  std::cout << std::endl << "############## Cut Statistics #############" << std::endl;
  std::cout << "nCuts: " << nCutsRed << std::endl;
  std::cout << "nTriv: " << nTrivRed << std::endl;
  std::cout << "nFilt: " << nFiltRed << std::endl;
  std::cout << "nSatu: " << nSatuRed << std::endl;
  std::cout << "nCutPerNode: " << (((double)nCutsRed) / this->nNodes) << std::endl;
  std::cout << "###########################################" << std::endl;
}

void PriCutManager::printRuntimes() {

  std::cout << std::endl << "#### Runtimes in microsecond ####" << std::endl;
  // std::cout << "Merge: " << mergeTime << std::endl;
  // std::cout << "Filter: " << filterTime << std::endl;
  // std::cout << "ProcTwo: " << procTwoTime << std::endl;
  // std::cout << "Compute: " << compTime << std::endl;
  // std::cout << "Schedule: " << scheduleTime << std::endl;
  std::cout << "Total: " << this->kcutTime << std::endl;
  std::cout << "#################################" << std::endl;
}

aig::Aig& PriCutManager::getAig() { return this->aig; }

int PriCutManager::getK() { return this->K; }

int PriCutManager::getC() { return this->C; }

int PriCutManager::getNWords() { return this->nWords; }

int PriCutManager::getNThreads() { return this->nThreads; }

bool PriCutManager::getCompTruthFlag() { return this->compTruth; }

long double PriCutManager::getKcutTime() { return this->kcutTime; }

void PriCutManager::setKcutTime(long double time) { this->kcutTime = time; }

PerThreadPriCutPool& PriCutManager::getPerThreadPriCutPool() {
  return this->perThreadPriCutPool;
}

PerThreadPriCutList& PriCutManager::getPerThreadPriCutList() {
  return this->perThreadPriCutList;
}

PerThreadAuxTruth& PriCutManager::getPerThreadAuxTruth() {
  return this->perThreadAuxTruth;
}

PriCut** PriCutManager::getNodePriCuts() { return this->nodePriCuts; }

Covering & PriCutManager::getCovering() { return this->covering; }

// ######################## BEGIN OPERATOR ######################## //
struct KPriCutOperator {

  PriCutManager& cutMan;

  KPriCutOperator(PriCutManager& cutMan) : cutMan(cutMan) {}

  void operator()(aig::GNode node, galois::UserContext<aig::GNode>& ctx) {

    aig::Aig& aig        = cutMan.getAig();
    aig::Graph& aigGraph = aig.getGraph();

    aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

    if (nodeData.type == aig::NodeType::AND) {

      // Touching outgoing neighobors to acquire their locks
      for (auto edge : aigGraph.out_edges(node)) {
      }

      // Combine Cuts
      auto inEdgeIt          = aigGraph.in_edge_begin(node);
      aig::GNode lhsNode     = aigGraph.getEdgeDst(inEdgeIt);
      aig::NodeData& lhsData = aigGraph.getData(lhsNode);
      bool lhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

      inEdgeIt++;
      aig::GNode rhsNode     = aigGraph.getEdgeDst(inEdgeIt);
      aig::NodeData& rhsData = aigGraph.getData(rhsNode);
      bool rhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

      PriCutPool* cutPool    = cutMan.getPerThreadPriCutPool().getLocal();
      PriCutList* cutList    = cutMan.getPerThreadPriCutList().getLocal();
      AuxTruth* auxTruth 		 = cutMan.getPerThreadAuxTruth().getLocal();

			//ctx.cautiousPoint();

      cutMan.computePriCuts(cutPool, cutList, auxTruth, nodeData.id, lhsData.id,
                         rhsData.id, lhsPolarity, rhsPolarity);

      // NextNodes
      for (auto edge : aigGraph.out_edges(node)) {
        aig::GNode nextNode = aigGraph.getEdgeDst(edge);
        aig::NodeData& nextNodeData =
            aigGraph.getData(nextNode, galois::MethodFlag::WRITE);
        nextNodeData.counter += 1;
        if (nextNodeData.counter == 2) {
          ctx.push(nextNode);
        }
      }
    } else {
      if (nodeData.type == aig::NodeType::PI) {
        // Touching outgoing neighobors to acquire their locks and their fanin
        // node's locks.
        for (auto outEdge : aigGraph.out_edges(node)) {
        }

				//ctx.cautiousPoint();

        // Set the trivial cut
        nodeData.counter      = 3;
        PriCutPool* cutPool   = cutMan.getPerThreadPriCutPool().getLocal();
        PriCut* trivialCut    = cutPool->getMemory();
        trivialCut->leaves[0] = nodeData.id;
        trivialCut->nLeaves++;
        trivialCut->sig = (1U << (nodeData.id % 31));
        if (cutMan.getCompTruthFlag()) {
          unsigned* cutTruth = cutMan.readTruth(trivialCut);
          for (int i = 0; i < cutMan.getNWords(); i++) {
            cutTruth[i] = 0xAAAAAAAA;
          }
        }
        cutMan.getNodePriCuts()[nodeData.id] = trivialCut;

        // Schedule next nodes
        for (auto edge : aigGraph.out_edges(node)) {
          aig::GNode nextNode = aigGraph.getEdgeDst(edge);
          aig::NodeData& nextNodeData =
              aigGraph.getData(nextNode, galois::MethodFlag::WRITE);
          nextNodeData.counter += 1;
          if (nextNodeData.counter == 2) {
            ctx.push(nextNode);
          }
        }
      }
    }
  }
};

void runKPriCutOperator(PriCutManager& cutMan) {

  galois::InsertBag<aig::GNode> workList;
  typedef galois::worklists::PerSocketChunkBag<500> DC_BAG;
  // typedef galois::worklists::PerSocketChunkFIFO< 200 > DC_FIFO;
  // typedef galois::worklists::PerSocketChunkLIFO< 200 > DC_LIFO;
  // typedef galois::worklists::PerThreadChunkFIFO< 200 > AC_FIFO;	

  for (auto pi : cutMan.getAig().getInputNodes()) {
    workList.push(pi);
  }

  // Galois Parallel Foreach
  galois::for_each(galois::iterate(workList.begin(), workList.end()),
                   KPriCutOperator(cutMan),
									 galois::wl<DC_BAG>(),
                   galois::loopname("KPriCutOperator"));


	cutMan.computeCovering();


	//galois::wl<galois::worklists::Deterministic<>>(),
	//galois::wl<DC_BAG>(),
	
}
// ######################## END OPERATOR ######################## //

} /* namespace algorithm */
