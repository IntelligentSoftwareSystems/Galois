#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "Element.hpp"
#include "../EquationSystem.h"

class Mesh;

class Node {
    private:
        int node = -1;
        Node *left = NULL;
        Node *right = NULL;
        Node *parent = NULL;
        std::vector<Element *> mergedElements;
        std::string production;
        std::vector<uint64_t> dofs;

        uint64_t dofsToElim;
        EquationSystem *system;

    public:
        int n_left = -1;
        int n_right = -1;

        std::vector<uint64_t> leftPlaces;
        std::vector<uint64_t> leftMergePlaces;

        std::vector<uint64_t> rightPlaces;
        std::vector<uint64_t> rightMergePlaces;

        Node(int num) : node(num) {}
        ~Node() {
            delete this->system;
        }

        void setLeft(Node *left);
        void setRight(Node *right);
        void setParent(Node *parent);

        Node *getLeft();
        Node *getRight();
        Node *getParent();

        void addElement (Element *e);
        std::vector<Element *> &getElements();
        void clearElements();

        void addDof(uint64_t dof);
        std::vector<uint64_t> &getDofs();
        void clearDofs();

        int getId();

        void setDofsToElim(uint64_t dofs);
        uint64_t getDofsToElim();

        void allocateSystem();

        void setProduction(std::string &prodname);
        std::string &getProduction();

        void (*mergeProduction)(double **matrixA, double *rhsA,
                                double **matrixB, double *rhsB,
                                double **matrixOut, double *rhsOut);

        void (*preprocessProduction)(double **matrixIn, double *rhsIn,
                                     double **matrixOut, double *rhsOut);
        void fillin();
        void merge();
        void eliminate();
        void bs();
        
        static bool isNeighbour (Node *node, Node *parent);
        static bool isNeighbour (Element *element1, Element *element2);
        int getNumberOfNeighbours(std::vector<Element *> * allElements);
        
        void rebuildElements();
        
        /*DEBUG*/
        
        int treeSize();
        
        /*END OF DEBUG*/
};

#endif // NODE_HPP
