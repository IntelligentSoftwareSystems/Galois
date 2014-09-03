#include "Analysis.hpp"
#include <algorithm>

using namespace std;

// todo: rotator
/*
int Analysis::treeHeight(Node *root)
{
    if (root == NULL) {
        return 0;
    }
    else {
        return std::max(Analysis::treeHeight(root->getLeft()), Analysis::treeHeight(root->getRight()))+1;
    }
}
*/
/*
int Analysis::rotate(Node *root, Node *parent)
{
    // two nodes are neighbours if the intersection of sets of DOFs is not empty.
    auto isNeighbour = [] (Node *node, Node *parent) {
        std::set<uint64_t> nodeDofs(node->getDofs().cbegin(), node->getDofs().cend());
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

        return false; // no common elements => no neighbourhood
    };

    int l=0;
    int r=0;

    if (root->getLeft() != NULL) {
        l = Analysis::rotate(root->getLeft(), root);
    }
    if (root->getRight() != NULL) {
        r = Analysis::rotate(root->getRight(), root);
    }

    int h=r-l;

    if (h==1 || h==-1 || h==0) {
        return h; //no rotation
    }

    if (h>=2)
    {//we need to perform some rotations to the left
     //  child=dziecko root->right które jest sąsiadem parent
     //  (if there are two such sons, we pick up the one with smaller
//   number of connected neighbors - with shorter neighbors)

        //
        Node *child;

        bool childIsRight = isNeighbour(root->getRight()->getRight(), parent);
        bool childIsLeft = isNeighbour(root->getRight()->getLeft(), parent);

        if (childIsRight && childIsLeft) {
            int l_length = Analysis::treeHeight(root->getRight()->getLeft());
            int r_length = Analysis::treeHeight(root->getRight()->getRight());
            if (l_length > r_length) {
                child = root->getRight()->getRight();
            } else {
                child = root->getRight()->getLeft();
            }
        }

        if ((childIsRight && r<=0) ||
            (childIsLeft && r>=0))
        {
            Node *T, *t;
            //    if parent!=NULL then make root->right son of parent
            //else T=root->right
            //t=root->left

            if (parent != NULL) {
                parent->setRight(root->getRight());
            } else {
                T = root->getRight();
            }

            t = root->getLeft();

            root->getRight()->setLeft(root);
            root->setRight(child);

            // WTF!? - podmieniamy roota?
            if (parent != NULL) {
                //merge_list(parent->left,parent->right)
                parent->set
            } else {

            }
            //      else merge_list(T->left,T->right)

            // end of left rotation
        }

//    make root left son of root->right
//    (this step may require exchange of left son
//     with right son for root->right)
//    make child right son of root
//    if parent!=NULL merge_list(parent->left,parent->right)
//      else merge_list(T->left,T->right)
//  }//end of rotation to the left
//  else //double rotation
//  {// lower level rotation to the right
//   (rotation at root->right)
//    parent1=root
//    root1=root->right
//    if parent1!=NULL then make root1->left son of parent
//    t1=root1->right
//    child1=son of root1->left being neighbor of t
//    (if there are two such sons, we pick up the one with
//     smaller number of connected neighbors)
//    make root1 right son of root1->left
//    (this step may require exchange of left son
//     with right son of root1->left)
//    make child1 left son of root1

//    //rotation on higher level in left
//     if parent!=NULL then make root->right son of parent
//       else T=root->right
//     t=root->left
//     child=son of root->right, being neighbor of t
//     make root left son of root->right
//     (this step may require exchange of left son
//      with right son for root->right)
//     make child right son of root
//  }//end of double rotations
//}// if (h>=2)
//else if (h<=-2) then
//{//we need to perform some rotations to the right
//if (child==root->left->right) && r<=0) ||
//         (child==root->left->left) && r>=0) then
//{//rotation to the right
//(Here we perform the same actions as for the rotations
// to the left, but in the symmetric way)
//}

    }
    return h;
}
*/
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
//        node->leftPlaces = new uint64_t[node->getLeft()->getDofs().size() - node->getLeft()->getDofsToElim()];
//        node->rightPlaces = new uint64_t[node->getRight()->getDofs().size() - node->getRight()->getDofsToElim()];

        map<uint64_t, uint64_t> reverseMap;
/*
        for (int i=0; i<node->getDofs().size(); ++i) {
            reverseMap[node->getDofs()[i]] = i;
        }

        for (int i=node->getLeft()->getDofsToElim(); i<node->getLeft()->getDofs().size(); ++i) {
            node->leftPlaces[i-node->getLeft()->getDofsToElim()] = reverseMap[node->getLeft()->getDofs()[i]];
        }

        for (int i=node->getRight()->getDofsToElim(); i<node->getRight()->getDofs().size(); ++i) {
            node->rightPlaces[i-node->getRight()->getDofsToElim()] = reverseMap[node->getRight()->getDofs()[i]];
        }
*/
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
            return tuple<edge, uint64_t>(edge(v1,vertex(2*x2-x1, y1)), 2);
        }
        if (levelEdges[level-1].count(edge(vertex(2*x1-x2, y1), v2))) {
            return tuple<edge, uint64_t>(edge(vertex(2*x1-x2, y1), v2), 1);
        }
        return tuple<edge, uint64_t>(e, 0);
    } else {
        if (levelEdges[level-1].count(edge(v1,vertex(x1, 2*y2-y1)))) {
            return tuple<edge, uint64_t>(edge(v1,vertex(x1, 2*y2-y1)), 2);
        }
        if (levelEdges[level-1].count(edge(vertex(x1, 2*y1-y2), v2))) {
            return tuple<edge, uint64_t>(edge(vertex(x1, 2*y1-y2), v2), 1);
        }
        return tuple<edge, uint64_t>(e, 0);
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

    tuple<edge, uint64_t> ve1 = Analysis::parentEdge(e1, levelVertices, levelEdges, level);
    tuple<edge, uint64_t> ve2 = Analysis::parentEdge(e2, levelVertices, levelEdges, level);
    tuple<edge, uint64_t> ve3 = Analysis::parentEdge(e3, levelVertices, levelEdges, level);
    tuple<edge, uint64_t> ve4 = Analysis::parentEdge(e4, levelVertices, levelEdges, level);


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

    auto compare = [](Element *e1, Element *e2) {
        if (e1->k > e2->k) {
            return false;
        } else if (e1->k < e2->k) {
            return true;
        }

        if (e1->l < e2->l) {
            return true;
        } else if (e1->l > e2->l) {
            return false;
        }
        return false;
    };

    std::sort(mesh->getElements().begin(), mesh->getElements().end(), compare);

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

void Analysis::debugNode(Node *n)
{
    printf("Node: %d\n", n->getId());
    printf("  dofs: ");
    for (uint64_t dof : n->getDofs()) {
        printf("%lu ", dof);
    }
    printf("\n");
}

void Analysis::printTree(Node *n)
{
    printf("Node id: %d ", n->getId());
    printf("[");
    for (uint64_t dof : n->getDofs()) {
        printf("%lu, ", dof);
    }
    printf("]");
    printf(" elim: %d\n", n->getDofsToElim());

    if(n->getLeft() != NULL && n->getRight() != NULL) {
        printTree(n->getLeft());
        printTree(n->getRight());
    }

}

void Analysis::printElement(Element *e)
{
    printf("E[%d,%d] at %d x %d -> %d x %d = [", e->k, e->l, e->x1, e->y1, e->x2, e->y2);
    for (uint64_t dof : e->dofs) {
        printf("%d, ", dof);
    }
    printf("]\n");
}
