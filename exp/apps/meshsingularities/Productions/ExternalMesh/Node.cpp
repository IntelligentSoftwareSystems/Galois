#include "Node.hpp"

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

void Node::eliminate()
{
    if (left != NULL && right != NULL) {
        this->mergeProduction(left->system->matrix, left->system->rhs,
                              right->system->matrix, right->system->rhs,
                              this->system->matrix, this->system->rhs);
        this->system->eliminate(this->getDofsToElim());
    } else {
        //this->preprocessProduction();
    }
    this->system->eliminate(this->getDofsToElim());
}

void Node::bs()
{
    this->system->backwardSubstitute(this->getDofsToElim());
}
