#include "Analysis.hpp"
using namespace std;

// todo: rotator

void Analysis::findLeftRotationChild(Node ** _parent, Node ** _root, bool child1, bool child2, Node ** _child, Node ** _otherChild, Mesh * mesh){
    Node * parent = (*_parent);
    Node * root = (*_root);
    
    Node * child;
    Node * otherChild;
    std::vector<Element *> allElements = mesh->getElements();
    
    if (child1){
        if (child2){
            if (root->getRight()->getLeft()->getNumberOfNeighbours(&allElements) < root->getRight()->getRight()->getNumberOfNeighbours(&allElements)){
                child = root->getRight()->getLeft();
                otherChild = root->getRight()->getRight();
            } else {
                child = root->getRight()->getRight();
                otherChild = root->getRight()->getLeft();
            }
        } else{
            child = root->getRight()->getLeft();
            otherChild = root->getRight()->getRight();
        }
    } else{
        child = root->getRight()->getRight();
        otherChild = root->getRight()->getLeft();
    }
    
    (*_child) = child;
    (*_otherChild) = otherChild;
}


void Analysis::findRightRotationChild(Node ** _parent, Node ** _root, bool child1, bool child2, Node ** _child, Node ** _otherChild, Mesh * mesh){
    Node * parent = (*_parent);
    Node * root = (*_root);
    
    Node * child;
    Node * otherChild;
    std::vector<Element *> allElements = mesh->getElements();
    
    if (child1){
        if (child2){
            if (root->getLeft()->getLeft()->getNumberOfNeighbours(&allElements) < root->getLeft()->getRight()->getNumberOfNeighbours(&allElements)){
                child = root->getLeft()->getLeft();
                otherChild = root->getLeft()->getRight();
            } else {
                child = root->getLeft()->getRight();
                otherChild = root->getLeft()->getLeft();
            }
        } else{
            child = root->getLeft()->getLeft();
            otherChild = root->getLeft()->getRight();
        }
    } else{
        child = root->getLeft()->getRight();
        otherChild = root->getLeft()->getLeft();
    }
    
    (*_child) = child;
    (*_otherChild) = otherChild;
}


Node * Analysis::leftRotation(Node ** _parent, Node ** _root, bool child1, bool child2, Node ** _child, Node ** _otherChild){
    
    Node * parent = (*_parent);
    Node * root = (*_root);
    Node * child = (*_child);
    Node * otherChild = (*_otherChild);
    
    if (parent != NULL){
        if (parent->getLeft() == root){
            parent->setLeft(root->getRight());
        } else {
            parent->setRight(root->getRight());
        }
    }
    Node * T = root->getRight();
    Node * t = root->getLeft();
    root->setRight(t);
    root->setLeft(child);
    T->setRight(otherChild);
    T->setLeft(root);
    
    T->rebuildElements();
    root->rebuildElements();
    
    /*std::set<uint64_t> *parentDofs;
    if (parent != NULL) {
        parentDofs = new set<uint64_t>(parent->getDofs().cbegin(), parent->getDofs().cend());
    } else {
        parentDofs = new set<uint64_t>();
    }
    Analysis::nodeAnaliser(T, parentDofs);*/
    
    return T;
}


Node * Analysis::rightRotation(Node ** _parent, Node ** _root, bool child1, bool child2, Node ** _child, Node ** _otherChild){
    
    Node * parent = (*_parent);
    Node * root = (*_root);
    Node * child = (*_child);
    Node * otherChild = (*_otherChild);
    
    if (parent != NULL){
        if (parent->getLeft() == root){
            parent->setLeft(root->getLeft());
        } else {
            parent->setRight(root->getLeft());
        }
    }
    Node * T = root->getLeft();
    Node * t = root->getRight();
    root->setLeft(t);
    root->setRight(child);
    T->setLeft(otherChild);
    T->setRight(root);
    
    T->rebuildElements();
    root->rebuildElements();
    
    /*std::set<uint64_t> *parentDofs;
    if (parent != NULL) {
        parentDofs = new set<uint64_t>(parent->getDofs().cbegin(), parent->getDofs().cend());
    } else {
        parentDofs = new set<uint64_t>();
    }
    Analysis::nodeAnaliser(T, parentDofs);*/
    
    return T;
}

int Analysis::rotate(Node * root, Node * parent, Mesh * mesh){
    
    int l,r,h;
    
    if (root->getLeft() != NULL) {
        l = Analysis::rotate(root->getLeft(), root, mesh);
    } else {
        l=0;
    }
    
    if (root->getRight() != NULL) {
        r = Analysis::rotate(root->getRight(), root, mesh);
    } else {
        r=0;
    }
    h=r-l;

    if ( (h==1) || (h==-1) || (h==0) ) { //no rotation
        return h;
    }

    Node * T = NULL;
    
    if (h>=2) {  //we need to perform some rotations to the left
        bool child1 = Node::isNeighbour(root->getRight()->getLeft(), parent);
        bool child2 = Node::isNeighbour(root->getRight()->getRight(), parent);
        Node * child;
        Node * otherChild;
        findLeftRotationChild(&parent, &root, child1, child2, &child, &otherChild, mesh);
        
        if (((child == root->getRight()->getRight()) && (r<=0)) || ((child == root->getRight()->getLeft()) && (r>=0))) {//rotation to the left
            T = leftRotation(&parent, &root, child1, child2, &child, &otherChild);
        }//end of rotation to the left
        
        else //double rotation
        {// lower level rotation to the right
            //   (rotation at root->right)
            Node * parent1 = root;
            Node * root1 = root->getRight();
            bool nchild1 = Node::isNeighbour(root1->getLeft()->getLeft(), parent);
            bool nchild2 = Node::isNeighbour(root1->getLeft()->getRight(), parent);
            Node * nchild;
            Node * notherChild;
            
            findRightRotationChild(&parent1, &root1, nchild1, nchild2, &nchild, &notherChild, mesh);
            rightRotation(&parent1, &root1, nchild1, nchild2, &nchild, &notherChild);
            
            //rotation on higher level in left
            child1 = Node::isNeighbour(root->getRight()->getLeft(), parent);
            child2 = Node::isNeighbour(root->getRight()->getRight(), parent);
            findLeftRotationChild(&parent, &root, child1, child2, &child, &otherChild, mesh);
            T = leftRotation(&parent, &root, child1, child2, &child, &otherChild);
        }//end of double rotations
    } else {// end if (h>=2)
    // the same, but other direction rotation
        bool child1 = Node::isNeighbour(root->getLeft()->getLeft(), parent);
        bool child2 = Node::isNeighbour(root->getLeft()->getRight(), parent);
        Node * child;
        Node * otherChild;
        findRightRotationChild(&parent, &root, child1, child2, &child, &otherChild, mesh);
        
        if (((child == root->getLeft()->getRight()) && (r<=0)) || ((child == root->getLeft()->getLeft()) && (r>=0))) {//rotation to the left
            T = rightRotation(&parent, &root, child1, child2, &child, &otherChild);
        }//end of rotation to the left
        
        else //double rotation
        {// lower level rotation to the left
            //   (rotation at root->left)
            Node * parent1 = root;
            Node * root1 = root->getLeft();
            bool nchild1 = Node::isNeighbour(root1->getRight()->getLeft(), parent);
            bool nchild2 = Node::isNeighbour(root1->getRight()->getRight(), parent);
            Node * nchild;
            Node * notherChild;
            
            findLeftRotationChild(&parent1, &root1, nchild1, nchild2, &nchild, &notherChild, mesh);
            leftRotation(&parent1, &root1, nchild1, nchild2, &nchild, &notherChild);
            
            //rotation on higher level in right
            child1 = Node::isNeighbour(root->getLeft()->getLeft(), parent);
            child2 = Node::isNeighbour(root->getLeft()->getRight(), parent);
            findRightRotationChild(&parent, &root, child1, child2, &child, &otherChild, mesh);
            T = rightRotation(&parent, &root, child1, child2, &child, &otherChild);
        }//end of double rotations
    }
    
    if (parent == NULL){
        mesh->setRootNode(T);
    }
    
    return 0;
}


void Analysis::nodeAnaliser(Node *node, set<uint64_t> *parent)
{
    auto getAllDOFs = [] (Node *n) {
        set<uint64_t> *dofs = new set<uint64_t>();
        for (Element *e : n->getElements()) {
            for (uint64_t dof : e->dofs)
                dofs->insert(dof);
        }
        return dofs;
    };

    set<uint64_t> *common;

    if (node->getLeft() != NULL && node->getRight() != NULL) {
        set<uint64_t> *lDofs = getAllDOFs(node->getLeft());
        set<uint64_t> *rDofs = getAllDOFs(node->getRight());

        common = new set<uint64_t>;
        std::set_intersection(lDofs->begin(), lDofs->end(),
                              rDofs->begin(), rDofs->end(),
                              std::inserter(*common, common->begin()));

        delete (lDofs);
        delete (rDofs);

        for (auto p = parent->cbegin(); p!=parent->cend(); ++p) {
            common->insert(*p);
        }

        Analysis::nodeAnaliser(node->getLeft(), common);
        Analysis::nodeAnaliser(node->getRight(), common);

    } else {
        common = getAllDOFs(node);
    }
    for (uint64_t dof : *common) {
        if (!parent->count(dof)) {
            node->addDof(dof);
        }
    }

    node->setDofsToElim(node->getDofs().size());

    for (uint64_t dof : *common) {
        if (parent->count(dof)) {
            node->addDof(dof);
        }
    }


//    printf("Node (%-2d): ", node->getId());
//    for (int dof : node->getDofs()) {
//        printf("%d ", dof);
//    }
//    printf("\n");

    delete common;
}

void Analysis::doAnalise(Mesh *mesh)
{
    Node *root = mesh->getRootNode();
    std::set<uint64_t> *parent = new set<uint64_t>();
    Analysis::nodeAnaliser(root, parent);
    Analysis::mergeAnaliser(root);

    delete parent;
}

void Analysis::mergeAnaliser(Node *node)
{
    if (node->getLeft() != NULL && node->getRight() != NULL) {
        std::set<uint64_t> leftNodes;
        std::set<uint64_t> rightNodes;
        std::set<uint64_t> mergeNodes;

        for (uint64_t i=node->getLeft()->getDofsToElim(); i<node->getLeft()->getDofs().size(); ++i) {
            leftNodes.insert(node->getLeft()->getDofs()[i]);
        }

        for (uint64_t i=node->getRight()->getDofsToElim(); i<node->getRight()->getDofs().size(); ++i) {
            rightNodes.insert(node->getRight()->getDofs()[i]);
        }

        for (uint64_t i=0; i<node->getDofs().size(); ++i) {
            mergeNodes.insert(node->getDofs()[i]);
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

        for (uint64_t elem : lCommonNodes) {
            for (int i=node->getLeft()->getDofsToElim(); i<node->getLeft()->getDofs().size(); ++i) {
                if (node->getLeft()->getDofs()[i] == elem) {
                    node->leftPlaces.push_back(i);
                    break;
                }
            }

            for (int i=0; i<node->getDofs().size(); ++i) {
                if (node->getDofs()[i] == elem) {
                    node->leftMergePlaces.push_back(i);
                    break;
                }
            }
        }


        for (uint64_t elem : rCommonNodes) {
            for (int i=node->getRight()->getDofsToElim(); i<node->getRight()->getDofs().size(); ++i) {
                if (node->getRight()->getDofs()[i] == elem) {
                    node->rightPlaces.push_back(i);
                    break;
                }
            }

            for (int i=0; i<node->getDofs().size(); ++i) {
                if (node->getDofs()[i] == elem) {
                    node->rightMergePlaces.push_back(i);
                    break;
                }
            }
        }
        Analysis::mergeAnaliser(node->getLeft());
        Analysis::mergeAnaliser(node->getRight());

    }
}

tuple<edge, uint64_t> Analysis::parentEdge(edge e,
                                          std::map<uint64_t, std::map<vertex, uint64_t> > &levelVertices,
                                          std::map<uint64_t, std::map<edge, uint64_t> > &levelEdges,
                                          uint64_t level)
{
    // this function returns either parent egde (if exists)
    // or the specified edge if there is no parent edge
    vertex &v1 = std::get<0>(e);
    vertex &v2 = std::get<1>(e);
    uint64_t x1 = std::get<0>(v1);
    uint64_t y1 = std::get<1>(v1);
    uint64_t x2 = std::get<0>(v2);
    uint64_t y2 = std::get<1>(v2);

    // horizontal edge
    if (y1 == y2) {
        if (levelEdges[level-1].count(edge(v1,vertex(2*x2-x1, y1)))) {
            return tuple<edge, bool>(edge(v1,vertex(2*x2-x1, y1)), 2);
        }
        if (levelEdges[level-1].count(edge(vertex(2*x1-x2, y1), v2))) {
            return tuple<edge, bool>(edge(vertex(2*x1-x2, y1), v2), 1);
        }
        return tuple<edge, bool>(e, 0);
    } else {
        if (levelEdges[level-1].count(edge(v1,vertex(x1, 2*y2-y1)))) {
            return tuple<edge, bool>(edge(v1,vertex(x1, 2*y2-y1)), 2);
        }
        if (levelEdges[level-1].count(edge(vertex(x1, 2*y1-y2), v2))) {
            return tuple<edge, bool>(edge(vertex(x1, 2*y1-y2), v2), 1);
        }
        return tuple<edge, bool>(e, 0);
    }
}

void Analysis::enumerateElem(Mesh *mesh, Element *elem,
                             std::map<uint64_t, std::map<vertex, uint64_t> > &levelVertices,
                             std::map<uint64_t, std::map<edge, uint64_t> > &levelEdges,
                             uint64_t &n, uint64_t level)
{
    map<vertex, uint64_t> &vertices = levelVertices[level];
    map<edge, uint64_t> &edges = levelEdges[level];

    edge e1(vertex(elem->x1, elem->y1), vertex(elem->x2, elem->y1));
    edge e2(vertex(elem->x2, elem->y1), vertex(elem->x2, elem->y2));
    edge e3(vertex(elem->x1, elem->y2), vertex(elem->x2, elem->y2));
    edge e4(vertex(elem->x1, elem->y1), vertex(elem->x1, elem->y2));

    // for other layers we need to take into consideration also
    // h-adaptation and its influence on positions of DOF
    map<vertex, uint64_t> &parentVertices = levelVertices[level-1];
    map<edge, uint64_t> &parentEdges = levelEdges[level-1];

    tuple<edge, bool> ve1 = Analysis::parentEdge(e1, levelVertices, levelEdges, level);
    tuple<edge, bool> ve2 = Analysis::parentEdge(e2, levelVertices, levelEdges, level);
    tuple<edge, bool> ve3 = Analysis::parentEdge(e3, levelVertices, levelEdges, level);
    tuple<edge, bool> ve4 = Analysis::parentEdge(e4, levelVertices, levelEdges, level);


    vertex v1(std::min(std::get<0>(std::get<0>(std::get<0>(ve1))),
                       std::get<0>(std::get<0>(std::get<0>(ve4)))),
              std::min(std::get<1>(std::get<0>(std::get<0>(ve1))),
                       std::get<1>(std::get<0>(std::get<0>(ve4)))));
    vertex v2(std::max(std::get<0>(std::get<1>(std::get<0>(ve1))),
                       std::get<0>(std::get<0>(std::get<0>(ve2)))),
              std::min(std::get<1>(std::get<1>(std::get<0>(ve1))),
                       std::get<1>(std::get<0>(std::get<0>(ve2)))));
    vertex v3(std::max(std::get<0>(std::get<1>(std::get<0>(ve2))),
                       std::get<0>(std::get<1>(std::get<0>(ve3)))),
              std::max(std::get<1>(std::get<1>(std::get<0>(ve2))),
                       std::get<1>(std::get<1>(std::get<0>(ve3)))));
    vertex v4(std::min(std::get<0>(std::get<0>(std::get<0>(ve3))),
                       std::get<0>(std::get<1>(std::get<0>(ve4)))),
              std::max(std::get<1>(std::get<0>(std::get<0>(ve3))),
                       std::get<1>(std::get<1>(std::get<0>(ve4)))));

    auto add_vertex = [&] (vertex &v) {
        if (parentVertices.count(v)) {
            vertices[v] = parentVertices[v];
        } else {
            if (!vertices.count(v)) {
                vertices[v] = n++;
            }
        }
        elem->dofs.push_back(vertices[v]);
    };

    auto add_edge = [&] (edge &e) {
        if (parentEdges.count(e)) {
            edges[e] = parentEdges[e];
        } else {
            if (!edges.count(e)) {
                edges[e] = n;
                n += mesh->getPolynomial()-1;
            }
        }
        for (uint64_t i=0; i<(mesh->getPolynomial()-1); ++i) {
            elem->dofs.push_back(edges[e]+i);
        }
    };

    add_vertex(v1);
    add_vertex(v2);
    add_vertex(v3);
    add_vertex(v4);

    add_edge(std::get<0>(ve1));
    add_edge(std::get<0>(ve2));
    add_edge(std::get<0>(ve3));
    add_edge(std::get<0>(ve4));

    for (uint64_t i=0; i<(mesh->getPolynomial()-1)*(mesh->getPolynomial()-1); ++i) {
        elem->dofs.push_back(n+i);
    }
    n += (mesh->getPolynomial()-1)*(mesh->getPolynomial()-1);
}

void Analysis::enumerateElem1(Mesh *mesh, Element *elem,
                             map<uint64_t, map<vertex, uint64_t>> &levelVertices,
                             map<uint64_t, map<edge, uint64_t>> &levelEdges,
                             uint64_t &n)
{
    map<vertex, uint64_t> &vertices = levelVertices[1];
    map<edge, uint64_t> &edges = levelEdges[1];

    vertex v1(elem->x1, elem->y1);
    vertex v2(elem->x2, elem->y1);
    vertex v3(elem->x2, elem->y2);
    vertex v4(elem->x1, elem->y2);

    edge e1(v1, v2);
    edge e2(v2, v3);
    edge e3(v4, v3);
    edge e4(v1, v4);

    auto add_vertex = [&] (vertex &v) {
        if (!vertices.count(v)) {
            vertices[v] = n++;
        }
        elem->dofs.push_back(vertices[v]);
    };

    auto add_edge = [&] (edge &e) {
        if (!edges.count(e)) {
            edges[e] = n;
            n += mesh->getPolynomial()-1;
        }
        for (uint64_t i=0; i<mesh->getPolynomial()-1; ++i) {
            elem->dofs.push_back(edges[e]+i);
        }
    };

    // vertices
    add_vertex(v1);
    add_vertex(v2);
    add_vertex(v3);
    add_vertex(v4);

    // edges
    add_edge(e1);
    add_edge(e2);
    add_edge(e3);
    add_edge(e4);
    // in 2-dimensional space the faces do not overlap
    for (uint64_t i=0; i<(mesh->getPolynomial()-1)*(mesh->getPolynomial()-1); ++i) {
        elem->dofs.push_back(n+i);
    }
    n += (mesh->getPolynomial()-1)*(mesh->getPolynomial()-1);
}

void Analysis::enumerateDOF(Mesh *mesh)
{
    map<uint64_t, vector<Element*>> elementMap;
    set<uint64_t> levels;

    map<uint64_t, map<vertex, uint64_t>> levelVertices;
    map<uint64_t, map<edge, uint64_t>> levelEdges;

    uint64_t n = 1;

    // now, we have level plan for mesh
    for (Element *e : mesh->getElements()) {
        levels.insert(e->k);
        elementMap[e->k].push_back(e);
    }

    // implementation assumes that the neighbours may vary on one level only
    for (uint64_t level : levels) {
        vector<Element *> elems = elementMap[level];
        // on the first layer we do not need to care about adaptation
        for (Element *elem : elems) {
            if (level == 1) {
                Analysis::enumerateElem1(mesh, elem, levelVertices, levelEdges, n);
            } else {
                Analysis::enumerateElem(mesh, elem, levelVertices, levelEdges, n, level);
            }
        }
    }
    mesh->setDofs(n-1);
}
