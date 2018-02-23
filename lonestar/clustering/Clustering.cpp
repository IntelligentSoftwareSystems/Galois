/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

// clang-format off
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"

#include "Lonestar/BoilerPlate.h"

#include <map>
#include <string>
#include <sstream>
#include <limits>
#include <fstream>
#include <set>
#include <iostream>
#include <limits>
#include <set>
#include <stdlib.h>

#include "LeafNode.h"
#include "NodeWrapper.h"
#include "KdTree.h"
#include "ClusterNode.h"
#include <stdlib.h>
#include <sys/time.h>
// clang-format on

namespace cll = llvm::cl;

static const char* name = "Unordered Agglomerative Clustering";
static const char* desc =
    "Clusters data points using the well-known data-mining algorithm";
static const char* url = "agglomerative_clustering";

static cll::opt<int> numPoints("numPoints", cll::desc("Number of Points"),
                               cll::init(1000));
static cll::opt<bool> parallel("parallel",
                               cll::desc("Run the parallel Galoised version?"),
                               cll::init(true));

#define DEBUG_CONSOLE 0

struct Clustering {

  std::vector<NodeWrapper*> lights;
  std::vector<double> coordinatesArray;

  std::vector<ClusterNode*> clusterArray;

  void genRandomLights(int numPoints) {

    double dirX = 0;
    double dirY = 0;
    double dirZ = 1;
    AbstractNode::setGlobalMultitime();
    AbstractNode::setGlobalNumReps();

    lights.reserve(numPoints);

    // generating random lights
    // TODO: use C++ random number generators
    for (int i = 0; i < numPoints; i++) {
      double x = ((double)rand()) / (numeric_limits<int>::max());
      double y = ((double)rand()) / (numeric_limits<int>::max());
      double z = ((double)rand()) / (numeric_limits<int>::max());

      Light l = LeafNode(x, y, z, dirX, dirY, dirZ);
      NodeWrapper* nw = new NodeWrapper(l);
      lights.push_back(nw);
    }
  }

  Clustering(int numPoints) {

    genRandomLights(numPoints);

    size_t tempSize = (1 << NodeWrapper::CONE_RECURSE_SIZE) + 1;
    coordinatesArray.resize(tempSize * 3);
    clusterArray.resize(tempSize);

  }

  ~Clustering(void) {
    for (auto* l: lights) {
      delete l;
    }
    lights.clear();
  }



  int findMatch(KdTree* tree, NodeWrapper* nodeA,
                galois::InsertBag<NodeWrapper*>*& newNodes,
                galois::InsertBag<NodeWrapper*>& allocs,
                GVector<double>* coordinatesArray,
                GVector<ClusterNode*>& clusterArray) {
    int addCounter = 0;
    if (tree->contains(*nodeA)) {

      NodeWrapper* nodeB = tree->findBestMatch((*nodeA));
      if (nodeB != NULL && tree->contains(*nodeB)) {
        NodeWrapper* nodeBMatch = tree->findBestMatch((*nodeB));
        if (nodeBMatch != NULL) {
          // cout<<" Match found "<<*nodeA<<" AND " << *nodeB<<endl;
          if (nodeA->equals(*nodeBMatch) && nodeA->equals(*nodeB) == false) {
            //   cout << " A   is " << *nodeA << " A-Closest=B::" << *nodeB
            //        << " B-Closest " << *nodeBMatch << endl;

            // Create a new node here.
            if (nodeA < nodeB) {
              NodeWrapper* newNode =
                  new NodeWrapper(*nodeA, *nodeB, coordinatesArray, clusterArray);
              newNodes->push(newNode);
              allocs.push(newNode);
              addCounter++;

            }
          } else {
            addCounter++;
            newNodes->push(nodeA);
            //   cout << " A   is " << *nodeA << " A-Closes=B:: " << *nodeB << " B
            //   - Closest " << *nodeBMatch << endl;
          }
        }
      } else {
        addCounter++;
        newNodes->push(nodeA);
      }
    }
    return addCounter;
  }
  template <typename V>
  void clusterGalois(V& lights) {

    int tempSize = (1 << NodeWrapper::CONE_RECURSE_SIZE) + 1;
    cout << "Temp size is " << tempSize << " coord. arr size should be "
         << tempSize * 3 << endl;
    std::vector<double>* coordinatesArray = new std::vector<double>(tempSize * 3);
    std::vector<ClusterNode*> clusterArray(tempSize);

#if DEBUG_CONSOLE

    cout << "Tree created " << *tree << endl;
    cout << "================================================================"
         << endl;
#endif

    GVector<NodeWrapper*> workList(0);
    KdTree::getAll(*tree, workList);
    size_t size = 0;

    galois::GAccumulator<size_t> addedNodes;
    galois::InsertBag<NodeWrapper*>* newNodes =

        new galois::InsertBag<NodeWrapper*>();
    galois::InsertBag<NodeWrapper*> allocs;

    galois::StatTimer T;
    T.start();
    while (true) {

      addedNodes.reset();

      using WL = galois::worklists::dChunkedFIFO<16>;

      // find matching
      galois::for_each(
          galois::iterate(workList),
          [&](NodeWrapper* nodeA, auto& lwl) {
            if (tree->contains(*nodeA)) {
              NodeWrapper* nodeB = tree->findBestMatch((*nodeA));
              if (nodeB != NULL && tree->contains(*nodeB)) {
                NodeWrapper* nodeBMatch = tree->findBestMatch((*nodeB));
                if (nodeBMatch != NULL) {
                  if (nodeA->equals(*nodeBMatch) &&
                      nodeA->equals(*nodeB) == false) {
                    // Create a new node here.
                    if (nodeA < nodeB) {
                      // galois::Allocator::
                      NodeWrapper* newNode = new NodeWrapper(
                          *nodeA, *nodeB, coordinatesArray, clusterArray);
                      newNodes->push(newNode);
                      allocs.push(newNode);
                      addedNodes += 1;
                    }
                  } else {
                    addedNodes += 1;
                    newNodes->push(nodeA);
                  }
                }
              } else {
                addedNodes += 1;
                newNodes->push(nodeA);
              }
            }
          },
          galois::wl<WL>(), galois::loopname("Main"));

      size += addedNodes.reduce();

      workList.clear();

      for (galois::InsertBag<NodeWrapper*>::iterator it    = newNodes->begin(),
                                                     itEnd = newNodes->end();
           it != itEnd; it++)
        workList.push_back(*it);
      if (size < 2)
        break;
      size = 0;

      galois::StatTimer Tcleanup("cleanup-time");
      Tcleanup.start();
      KdCell::cleanupTree(tree);
      Tcleanup.stop();

    newNodes->clear();
  }
  T.stop();

      galois::StatTimer TcreateTree("create-tree-time");
      TcreateTree.start();
      tree = (KdTree::createTree(workList));
      TcreateTree.stop();

      newNodes->clear_parallel();
    }
    T.stop();
  }

  void clusterSerial(GVector<LeafNode*>& lights) {

    int tempSize = (1 << NodeWrapper::CONE_RECURSE_SIZE) + 1;
    cout << "Temp size is " << tempSize << " coord. arr size should be "
         << tempSize * 3 << endl;

    std::vector<double>* coordinatesArray = new std::vector<double>(tempSize * 3);
    std::vector<NodeWrapper*> initialWorklist(lights.size());
    std::vector<ClusterNode*> clusterArray(tempSize);

    for (unsigned int i = 0; i < lights.size(); i++) {
      NodeWrapper* nw    = new NodeWrapper(*(lights[i]));
      initialWorklist[i] = nw;
    }

    KdTree* tree = (KdTree::createTree(initialWorklist));
    //#if DEBUG_CONSOLE
    cout << "Tree created " << *tree << endl;
    cout << "================================================================"
         << endl;
    //#endif

    std::vector<NodeWrapper*> workList(0);
    KdTree::getAll(*tree, workList);

    // TODO: remove
    // for (unsigned int i = 0; i < workListOld.size(); i++) {
      // workList.push_back(workListOld[i]);
    // }

    std::vector<NodeWrapper*> newNodes;
    std::vector<NodeWrapper*> allocs;

    while (true) {
      while (workList.size() > 1) {
        cout << "===================Worklist size :: " << workList.size()
             << "===============" << endl;

        NodeWrapper* nodeA = workList.back();
        workList.pop_back();

        assert(tree->contains(*nodeA) && "Node not in tree?");
        auto* match = tree->findBestMatch(*nodeA);

        if (match != nullptr) {
          assert(tree->contains(*match) && "match not in tree?");

          auto* counterMatch = tree->findBestMatch(*match);
  GVector
          if (counterMatch != nullptr) {
            assert(tree->contains(*counterMatch) && "counterMatch not in tree?");

            if (counterMatch->equals(*nodeA)) {
              auto* ccMatch = tree->findBestMatch(counterMatch);
              assert(ccMatch && ccMatch->equals(*match) && "mismatch in best matches");

              NodeWrapper* clust  = new NodeWrapper(*nodeA, *nodeB, coordinatesArray, clusterArray);
              newNodes.push_back(clust);
              allocs.push_back(clust);

            } else {GVector
              newNodes.push_back(nodeA);
            }
          }
        } else {
          newNodes.push_back(nodeA);
        }

        // if (tree->contains(*nodeA)) {
          // NodeWrapper* nodeB = tree->findBestMatch((*nodeA));
          // if (nodeB != NULL && tree->contains(*nodeB)) {
            // NodeWrapper* nodeBMatch = tree->findBestMatch((*nodeB));
            // if (nodeBMatch != NULL) {
              // if (nodeA->equals(*nodeBMatch) && nodeA->equals(*nodeB) == false) {
                // if (nodeA < nodeB) {
                  // NodeWrapper* newNode = new NodeWrapper(
                      // *nodeA, *nodeB, coordinatesArray, clusterArray);
                  // newNodes.push_back(newNode);
                  // allocs.push_back(newNode);
                // }
              // } else {
                // newNodes.push_back(nodeA);
              // }
            // }
          // } else {
            // newNodes.push_back(nodeA);
          // }
        // }
      }

      if (newNodes.size() < 2) {
        break;
      }

      workList.clear();
      for (NodeWrapper* nw: newNodes) {
        workList.push_back(nw);
      }

      cout << "Newly added" << newNodes.size() << endl;

      KdCell::cleanupTree(tree);
      tree = (KdTree::createTree(workList));
      newNodes.clear();
      
    }

#if DEBUG_CONSOLE
    cout << "================================================================"
         << endl
         << *tree << endl;
#endif

    //!!!!!!!!!!!!!!!!!!!Cleanup
    KdCell::cleanupTree(tree);
    AbstractNode::cleanup();

    for (unsigned int i = 0; i < lights.size(); i++) {
      NodeWrapper* nw = initialWorklist[i];
      delete nw;
    }

    for (auto* a: allocs) {
      delete a;
    }

    delete coordinatesArray;

    return;
  }



};


int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  cout << "Starting Clustering app...[" << numPoints << "]" << endl;
  // Initializing...

  std::vector<LeafNode> lights;
  lights.reserve(numPoints);
  genRandomLights(lights, numPoints);

  cout << "Running the " << (parallel ? "parallel" : "serial") << " version\n ";
  if (!parallel) {
    clusterSerial(lights);
  } else {
    clusterGalois(lights);
  }

  cout << "Ending Clustering app...[" << lights->size() << "]" << endl;
  return 0;
}
