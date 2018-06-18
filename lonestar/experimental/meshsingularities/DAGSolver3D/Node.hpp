#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "Element.hpp"
#include "EquationSystem.hpp"

#include <set>

class Mesh;

class Node {
private:
  int node     = -1;
  Node* left   = NULL;
  Node* right  = NULL;
  Node* parent = NULL;
  std::vector<uint64_t>* meshSupernodes;
  std::vector<Element*> mergedElements;
  std::string production;
  std::vector<uint64_t> supernodes;
  std::vector<uint64_t> offsets;
  uint64_t supernodesToElim;

public:
  EquationSystem*
      system; // TODO: temporary, add friend class for serialization or sth.

  int n_left  = -1;
  int n_right = -1;

  int l = 0;
  int r = 0;

  std::vector<uint64_t> leftPlaces;
  std::vector<uint64_t> rightPlaces;

  Node() : node(0), system(NULL) {}

  Node(std::vector<uint64_t>* meshSupernodes, int num)
      : meshSupernodes(meshSupernodes), node(num), system(NULL) {}
  ~Node() { delete system; }

  void setLeft(Node* left);
  void setRight(Node* right);
  void setParent(Node* parent);

  Node* getLeft() const;
  Node* getRight() const;
  Node* getParent() const;

  void addElement(Element* e);
  std::vector<Element*>& getElements();
  void clearElements();

  void addSupernode(uint64_t supernode);
  std::vector<uint64_t>& getSupernodes();
  void clearSupernodes();
  void computeOffsets();

  int getId() const;

  void setSupernodesToElim(uint64_t dofs);
  uint64_t getSupernodesToElim() const;

  void allocateSystem(SolverMode mode);
  void deallocateSystem();

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

  unsigned long getSizeInMemory(bool recursive = true);
  unsigned long getFLOPs(bool recursive = true);
  unsigned long getMemoryRearrangements();

  /*DEBUG*/

  int treeSize();

  /*END OF DEBUG*/
};

#endif // NODE_HPP
