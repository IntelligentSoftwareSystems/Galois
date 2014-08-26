#include "Analysis.hpp"
using namespace std;

// todo: rotator

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
    delete parent;
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
