#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "Element.hpp"
#include "EquationSystem.h"

#include <set>

class Mesh;

class Node {
private:
  int node     = -1;
  Node* left   = NULL;
  Node* right  = NULL;
  Node* parent = NULL;
  std::vector<Element*> mergedElements;
  std::string production;
  std::vector<uint64_t> dofs;

  uint64_t dofsToElim;
  EquationSystem* system;

public:
  int n_left  = -1;
  int n_right = -1;

  int l = 0;
  int r = 0;

  uint64_t* leftPlaces  = NULL;
  uint64_t* rightPlaces = NULL;

  Node(int num) : node(num) {}
  ~Node() {
    delete system;
    if (leftPlaces != NULL)
      delete[] leftPlaces;
    if (rightPlaces != NULL)
      delete[] rightPlaces;
  }

  void setLeft(Node* left);
  void setRight(Node* right);
  void setParent(Node* parent);

  Node* getLeft() const;
  Node* getRight() const;
  Node* getParent() const;

  void addElement(Element* e);
  std::vector<Element*>& getElements();
  void clearElements();

  void addDof(uint64_t dof);
  std::vector<uint64_t>& getDofs();
  void clearDofs();

  int getId() const;

  void setDofsToElim(uint64_t dofs);
  uint64_t getDofsToElim() const;

  void allocateSystem(SolverMode mode);

  void setProduction(std::string& prodname);
  std::string& getProduction();

  void (*mergeProduction)(double** matrixA, double* rhsA, double** matrixB,
                          double* rhsB, double** matrixOut, double* rhsOut);

  void (*preprocessProduction)(double** matrixIn, double* rhsIn,
                               double** matrixOut, double* rhsOut);
  void fillin() const;
  void merge() const;
  void eliminate() const;
  void bs() const;

  unsigned long getSizeInMemory();
  unsigned long getFLOPs();
  unsigned long getMemoryRearrangements();

  static bool isNeighbour(Node* node, Node* parent);
  static bool isNeighbour(Element* element1, Element* element2);
  int getNumberOfNeighbours(std::vector<Element*>& allElements);

  void rebuildElements();

  /*DEBUG*/

  int treeSize();
  int isBalanced(bool* bal);
  bool isLooped(std::set<Node*>* n);

  /*END OF DEBUG*/
};

#endif // NODE_HPP
