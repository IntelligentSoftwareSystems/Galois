#include "Node.hpp"
#include <set>
#include <algorithm>
#include "Analysis.hpp"

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

void Node::clearElements(){
    this->mergedElements.clear();
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

void Node::clearDofs(){
    this->dofs.clear();
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


bool Node::isNeighbour (Node *node, Node *parent)
{
    if (parent == NULL){
        return true;
    }
    /*std::set<uint64_t> nodeDofs(node->getDofs().cbegin(), node->getDofs().cend());
    std::set<uint64_t> parentDofs(parent->getDofs().cbegin(), parent->getDofs().cend());

    if (nodeDofs.empty() || parentDofs.empty()) {
        return false;
    }
    
    auto nodeIt = nodeDofs.begin();
    auto nodeItEnd = nodeDofs.end();

    auto parentIt = parentDofs.begin();
    auto parentItEnd = parentDofs.end();

    // sets are internally sorted, so if the beginning of the one is greater
    // than end of other - we can finish
    if ((*parentIt > *nodeDofs.rbegin()) || (*nodeIt > *parentDofs.rbegin())) {
        return false;
    }

    while (nodeIt != nodeItEnd && parentIt != parentItEnd) {
        if (*nodeIt == *parentIt) {
            return true; // common element? yes, we are neighbours!
        } else if (*nodeIt > *parentIt) {
            ++parentIt;
        } else {
            ++nodeIt;
        }
    }
    return false; // no common elements => no neighbourhood*/
    
    /*for (Element * e1 : node->getElements()){
        for (Element * e2 : parent->getElements()){
            if (Node::isNeighbour(e1, e2)){
                return true;
            }
        }
    }*/
    
    auto getAllDOFs = [] (Node *n) {
        std::set<uint64_t> *dofs = new std::set<uint64_t> ();
        for (Element *e : n->getElements()) {
            for (uint64_t dof : e->dofs)
                dofs->insert(dof);
        }
        return dofs;
    };
    
    std::set<uint64_t> * nodeDofs;
    std::set<uint64_t> * parentDofs;
    
    nodeDofs = getAllDOFs(node);
    parentDofs = getAllDOFs(parent);
    
    if (nodeDofs->empty() || parentDofs->empty()) {
        return false;
    }
    
    auto nodeIt = nodeDofs->begin();
    auto nodeItEnd = nodeDofs->end();

    auto parentIt = parentDofs->begin();
    auto parentItEnd = parentDofs->end();

    // sets are internally sorted, so if the beginning of the one is greater
    // than end of other - we can finish
    if ((*parentIt > *nodeDofs->rbegin()) || (*nodeIt > *parentDofs->rbegin())) {
        return false;
    }

    while (nodeIt != nodeItEnd && parentIt != parentItEnd) {
        if (*nodeIt == *parentIt) {
            return true; // common element? yes, we are neighbours!
        } else if (*nodeIt > *parentIt) {
            ++parentIt;
        } else {
            ++nodeIt;
        }
    }
    
    delete nodeDofs;
    delete parentDofs;
    
    return false;
};


bool Node::isNeighbour (Element *element1, Element *element2)
{
    std::set<uint64_t> element1Dofs(element1->dofs.cbegin(), element1->dofs.cend());
    std::set<uint64_t> element2Dofs(element2->dofs.cbegin(), element2->dofs.cend());

    if (element1Dofs.empty() || element2Dofs.empty()) {
        return false;
    }
    
    auto element1It = element1Dofs.begin();
    auto element1ItEnd = element1Dofs.end();

    auto element2It = element2Dofs.begin();
    auto element2ItEnd = element2Dofs.end();

    // sets are internally sorted, so if the beginning of the one is greater
    // than end of other - we can finish
    if ((*element2It > *element1Dofs.rbegin()) || (*element1It > *element2Dofs.rbegin())) {
        return false;
    }

    while (element1It != element1ItEnd && element2It != element2ItEnd) {
        if (*element1It == *element2It) {
            return true; // common element? yes, we are neighbours!
        } else if (*element1It > *element2It) {
            ++element2It;
        } else {
            ++element1It;
        }
    }
    return false; // no common elements => no neighbourhood
};


int Node::getNumberOfNeighbours(std::vector<Element *> * allElements){
    int common = 0;
    for (Element * e1 : (*allElements)) {
        for (Element * e2 : this->mergedElements) {
            if (Node::isNeighbour(e1, e2)){
                common++;
            }
        }
    }
    common -= this->mergedElements.size();
    return common;
}

void Node::rebuildElements(){
    this->clearDofs();
    this->clearElements();
    
    for (Element * e : this->getLeft()->getElements()){
        this->addElement(e);
    }
    for (Element * e : this->getRight()->getElements()){
        this->addElement(e);
    }
}




/* DEBUG*/

int Node::treeSize(){
    int ret = 1;
    if (this->getLeft() != NULL){
        ret += this->getLeft()->treeSize();
    }
    if (this->getRight() != NULL){
        ret += this->getRight()->treeSize();
    }
    return ret;
}

/*END OF DEBUG*/
