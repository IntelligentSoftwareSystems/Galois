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

Node *Node::getLeft()
{
    return this->left;
}

Node *Node::getRight()
{
    return this->right;
}

Node *Node::getParent()
{
    return this->parent;
}

std::vector<Element *> &Node::getElements()
{
    return this->mergedElements;
}

int Node::getId()
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

uint64_t Node::getDofsToElim()
{
    return this->dofsToElim;
}

void Node::allocateSystem()
{
    this->system = new EquationSystem(this->getDofs().size());
}

void Node::fillin()
{
    int i, j;
    for (i=0; i<this->system->n; ++i) {
        this->system->matrix[i][i] = 1.0;
        for (j=i; j<this->system->n; ++j) {
            this->system->matrix[i][j] = 2.0;
        }
        this->system->rhs[i] = 0.5;
    }
}

void Node::merge()
{
    std::set<uint64_t> leftNodes;
    std::set<uint64_t> rightNodes;
    std::set<uint64_t> mergeNodes;

    for (uint64_t i=left->getDofsToElim(); i<left->getDofs().size(); ++i) {
        leftNodes.insert(left->dofs[i]);
    }

    for (uint64_t i=right->getDofsToElim(); i<right->getDofs().size(); ++i) {
        rightNodes.insert(right->dofs[i]);
    }

    for (uint64_t i=0; i<this->getDofs().size(); ++i) {
        mergeNodes.insert(this->dofs[i]);
    }

    std::vector<uint64_t> lCommonNodes(leftNodes.size());
    std::vector<uint64_t>::iterator itLeft;
    std::vector<uint64_t> rCommonNodes(rightNodes.size());
    std::vector<uint64_t>::iterator itRight;

    itLeft = std::set_intersection(leftNodes.begin(), leftNodes.end(),
                          mergeNodes.begin(), mergeNodes.end(),
                          lCommonNodes.begin());

    itRight = std::set_intersection(rightNodes.begin(), rightNodes.end(),
                          mergeNodes.begin(), mergeNodes.end(),
                          rCommonNodes.begin());

    lCommonNodes.resize(itLeft - lCommonNodes.begin());
    rCommonNodes.resize(itRight - rCommonNodes.begin());

    std::vector<uint64_t> leftPlaces;
    std::vector<uint64_t> leftMergePlaces;

    std::vector<uint64_t> rightPlaces;
    std::vector<uint64_t> rightMergePlaces;

    for (uint64_t elem : lCommonNodes) {
        for (int i=left->getDofsToElim(); i<left->getDofs().size(); ++i) {
            if (left->dofs[i] == elem) {
                leftPlaces.push_back(i);
                break;
            }
        }

        for (int i=0; i<this->getDofs().size(); ++i) {
            if (this->dofs[i] == elem) {
                leftMergePlaces.push_back(i);
                break;
            }
        }
    }


    for (uint64_t elem : rCommonNodes) {
        for (int i=right->getDofsToElim(); i<right->getDofs().size(); ++i) {
            if (right->dofs[i] == elem) {
                rightPlaces.push_back(i);
                break;
            }
        }

        for (int i=0; i<this->getDofs().size(); ++i) {
            if (this->dofs[i] == elem) {
                rightMergePlaces.push_back(i);
                break;
            }
        }
    }

    for (int i=0; i<leftPlaces.size(); ++i) {
        for (int j=0; j<leftPlaces.size(); ++j) {
            this->system->matrix[leftMergePlaces[i]][leftMergePlaces[j]] = left->system->matrix[leftPlaces[i]][leftPlaces[j]];
        }
        this->system->rhs[leftMergePlaces[i]] = left->system->rhs[leftPlaces[i]];
    }

    for (int i=0; i<rightPlaces.size(); ++i) {
        for (int j=0; j<rightPlaces.size(); ++j) {
            this->system->matrix[rightMergePlaces[i]][rightMergePlaces[j]] += right->system->matrix[rightPlaces[i]][rightPlaces[j]];
        }
        this->system->rhs[rightMergePlaces[i]] += right->system->rhs[rightPlaces[i]];
    }

}

void Node::eliminate()
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
    this->system->eliminate(this->getDofsToElim());
}

void Node::bs()
{
    this->system->backwardSubstitute(this->getDofsToElim());
}
