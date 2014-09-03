#include "Node.hpp"
#include <set>
#include <algorithm>

void Node::setLeft(Node *left)
{
    this->left = left;
}

void Node::setRight(Node *right)
{
    this->right = right;
}

void Node::setParent(Node *parent)
{
    this->parent = parent;
}

void Node::addElement(Element *e)
{
    this->mergedElements.push_back(e);
}

void Node::setProduction(std::string &prodname)
{
    this->production = prodname;
}

std::string &Node::getProduction()
{
    return this->production;
}

Node *Node::getLeft() const
{
    return this->left;
}

Node *Node::getRight() const
{
    return this->right;
}

Node *Node::getParent() const
{
    return this->parent;
}

std::vector<Element *> &Node::getElements()
{
    return this->mergedElements;
}

int Node::getId() const
{
    return this->node;
}

void Node::addDof(uint64_t dof)
{
    this->dofs.push_back(dof);
}

std::vector<uint64_t> &Node::getDofs()
{
    return this->dofs;
}

void Node::setDofsToElim(uint64_t dofs)
{
    this->dofsToElim = dofs;
}

uint64_t Node::getDofsToElim() const
{
    return this->dofsToElim;
}

void Node::allocateSystem()
{
    this->system = new EquationSystem(this->getDofs().size());
    //printf("Size: %d x %d\n", system->n, system->n);
}

void Node::fillin() const
{
    int i;
    for (i=0; i<this->system->n; ++i) {
        for (int j=0; j<this->system->n; ++j) {
            this->system->matrix[i][j] = i == j ? 1.0 : 0.0;
        }
        this->system->rhs[i] = 1.0;
    }
}

void Node::merge() const
{
    for (int i=getLeft()->getDofsToElim(); i<getLeft()->getDofs().size(); ++i) {
        for (int j=getLeft()->getDofsToElim(); j<getLeft()->getDofs().size(); ++j) {
            //system->matrix[leftPlaces[i-getLeft()->getDofsToElim()]][leftPlaces[j-getLeft()->getDofsToElim()]] =
            //        left->system->matrix[i][j];
        }
        //system->rhs[leftPlaces[i-getLeft()->getDofsToElim()]] = left->system->rhs[i];
    }

    for (int i=getRight()->getDofsToElim(); i<getRight()->getDofs().size(); ++i) {
        for (int j=getRight()->getDofsToElim(); j<getRight()->getDofs().size(); ++j) {
            //system->matrix[rightPlaces[i-getRight()->getDofsToElim()]][rightPlaces[j-getRight()->getDofsToElim()]] =
                   // right->system->matrix[i][j];
        }
        //system->rhs[rightPlaces[i-getRight()->getDofsToElim()]] = right->system->rhs[i];
    }
/*
    for (int i=0; i<leftPlaces.size(); ++i) {
        for (int j=0; j<leftPlaces.size(); ++j) {
            system->matrix[leftMergePlaces[i]][leftMergePlaces[j]] = left->system->matrix[leftPlaces[i]][leftPlaces[j]];
        }
        system->rhs[leftMergePlaces[i]] = left->system->rhs[leftPlaces[i]];
    }

    for (int i=0; i<rightPlaces.size(); ++i) {
        for (int j=0; j<rightPlaces.size(); ++j) {
            system->matrix[rightMergePlaces[i]][rightMergePlaces[j]] += right->system->matrix[rightPlaces[i]][rightPlaces[j]];
        }
        system->rhs[rightMergePlaces[i]] += right->system->rhs[rightPlaces[i]];
    }
*/
}

void Node::eliminate() const
{
    if (left != NULL && right != NULL) {
        //this->mergeProduction(left->system->matrix, left->system->rhs,
        //                      right->system->matrix, right->system->rhs,
        //                      this->system->matrix, this->system->rhs);
        this->merge();
    } else {
        //this->preprocessProduction();
        this->fillin();
    }
    system->eliminate(getDofsToElim());
}

void Node::bs() const
{
    //system->backwardSubstitute(this->getDofsToElim());
    /*for (int i=0; i<system->n; ++i) {
        if (fabs(system->rhs[i]-1.0) > 1e-8) {
            printf("WRONG SOLUTION - [%lu] %d: %lf\n", this->getId(), i, system->rhs[i]);
        }
    }*/
}
