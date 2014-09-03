#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "Element.hpp"
#include "../EquationSystem.h"

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

        uint64_t *leftPlaces;
        uint64_t *rightPlaces;

        Node(int num) : node(num) {}
        ~Node() {
            delete system;
            //delete leftPlaces;
            //delete rightPlaces;
        }

        void setLeft(Node *left);
        void setRight(Node *right);
        void setParent(Node *parent);

        Node *getLeft() const;
        Node *getRight() const;
        Node *getParent() const;

        void addElement (Element *e);
        std::vector<Element *> &getElements();

        void addDof(uint64_t dof);
        std::vector<uint64_t> &getDofs();

        int getId() const;

        void setDofsToElim(uint64_t dofs);
        uint64_t getDofsToElim() const;

        void allocateSystem();

        void setProduction(std::string &prodname);
        std::string &getProduction();

        void (*mergeProduction)(double **matrixA, double *rhsA,
                                double **matrixB, double *rhsB,
                                double **matrixOut, double *rhsOut);

        void (*preprocessProduction)(double **matrixIn, double *rhsIn,
                                     double **matrixOut, double *rhsOut);
        void fillin() const;
        void merge() const;
        void eliminate() const;
        void bs() const;
};

#endif // NODE_HPP
