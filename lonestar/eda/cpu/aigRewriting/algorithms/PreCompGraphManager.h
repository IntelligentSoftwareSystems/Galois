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

#ifndef PRECOMPGRAPHMANAGER_h_
#define PRECOMPGRAPHMANAGER_h_

#include "NPNManager.h"
#include <vector>

namespace algorithm {

typedef struct DecEdge_ {

  unsigned fCompl : 1; // the complemented bit
  unsigned Node : 30;  // the decomposition node pointed by the edge

} DecEdge;

typedef struct DecNode_ {

  DecEdge eEdge0; // the left child of the node
  DecEdge eEdge1; // the right child of the node
  // other info
  union {
    int iFunc;   // the literal of the node (AIG)
    void* pFunc; // the function of the node (BDD or AIG)
  };
  int id;
  unsigned Level : 14; // the level of this node in the global AIG
  // printing info
  unsigned fNodeOr : 1; // marks the original OR node
  unsigned fCompl0 : 1; // marks the original complemented edge
  unsigned fCompl1 : 1; // marks the original complemented edge
  // latch info
  unsigned nLat0 : 5; // the number of latches on the first edge
  unsigned nLat1 : 5; // the number of latches on the second edge
  unsigned nLat2 : 5; // the number of latches on the output edge

} DecNode;

class DecGraph {

private:
  bool fConst; // marks the constant 1 graph
  int nLeaves; // the number of leaves
  int nSize;   // the number of nodes (including the leaves)
  int nCap;    // the number of allocated nodes
  int idCounter;
  DecNode* pNodes; // the array of leaves and internal nodes
  DecEdge eRoot;   // the pointer to the topmost node

  DecNode* appendNode();

public:
  DecGraph();                                   // Create a Const graph
  DecGraph(int nLeaves);                        // Create a graph with nLeaves
  DecGraph(int iLeaf, int nLeaves, int fCompl); // Create a leaf graph
  ~DecGraph();

  DecEdge addAndNode(DecEdge eEdge0, DecEdge eEdge1);
  DecEdge addOrNode(DecEdge eEdge0, DecEdge eEdge1);
  DecEdge addXorNode(DecEdge eEdge0, DecEdge eEdge1, int Type);
  DecEdge createEdge(unsigned Node, unsigned fCompl);

  DecNode* getNodes();
  DecNode* getNode(int i);
  DecNode* getVar();
  DecEdge getRootEdge();
  void setRootEdge(DecEdge eRoot);
  int getLeaveNum();
  int getNodeNum();
  bool isConst();
  bool isVar();
  unsigned isComplement();

  int nodeToInt(DecNode* node);
  int varToInt();
  DecEdge intToEdge(unsigned Edge);
  unsigned edgeToInt(DecEdge eEdge);

  unsigned deriveTruth();
};

typedef struct ForestNode_ ForestNode;
struct ForestNode_ {

  int Id;     // ID
  int TravId; // traversal ID
  short nScore;
  short nGain;
  short nAdded;
  unsigned uTruth : 16; // truth table
  unsigned Volume : 8;  // volume
  unsigned Level : 6;   // level
  unsigned fUsed : 1;   // mark
  unsigned fExor : 1;   // mark
  ForestNode* p0;       // first child
  ForestNode* p1;       // second child
  ForestNode* pNext;    // next in the table
};

class PreCompGraphManager {

private:
  static const unsigned short aigSubgraphs[3562];

  NPNManager& npnManager;
  ForestNode* forest; // all the nodes
  std::vector<ForestNode*>
      table; // the hash table of nodes by their canonical form
  std::vector<std::vector<ForestNode*>>
      classes; // the nodes of the equivalence classes
  int forestSize;
  int nTravIds;

  ForestNode* addForestNode(ForestNode* p0, ForestNode* p1, int fExor,
                            int Level, int Volume);
  ForestNode* addForestVar(unsigned uTruth);
  void addForestNodeToTable(unsigned uTruth, ForestNode* node);

  int getForestNodeVolume(ForestNode* p0, ForestNode* p1);
  void getVolumeRec(ForestNode* node, int* volume);
  void incTravId();

  bool isForestNodeComplement(ForestNode* node);
  ForestNode* forestNodeRegular(ForestNode* node);
  ForestNode* forestNodeComplement(ForestNode* node);
  ForestNode* forestNodeComplementCond(ForestNode* node, int c);

  DecGraph* processNode(ForestNode* node);
  DecEdge processNodeRec(ForestNode* node, DecGraph* decGraph);

public:
  PreCompGraphManager(NPNManager& npnManager);
  ~PreCompGraphManager();

  void loadPreCompGraphFromArray();
  void processDecompositionGraphs();

  ForestNode* getForest();
  std::vector<ForestNode*>& getTable();
  std::vector<std::vector<ForestNode*>>& getClasses();
};

} /* namespace algorithm */

#endif /* PRECOMPGRAPHMANAGER_H_ */
