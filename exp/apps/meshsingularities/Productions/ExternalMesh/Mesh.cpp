#include "Mesh.hpp"

void Mesh::addElement(Element *e)
{
    this->elements.push_back(e);
}

void Mesh::addNode(Node *n)
{
    this->nodes.push_back(n);
}

Node *Mesh::getRootNode()
{
    return this->nodes[0];
}

std::vector<Element *> &Mesh::getElements()
{
    return this->elements;
}

int Mesh::getPolynomial()
{
    return this->polynomial;
}

void Mesh::setDofs(int dofs)
{
    this->dofs = dofs;
}

int Mesh::getDofs()
{
    return this->dofs;
}

bool Mesh::saveToFile(const char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL) {
        return false;
    }

    fprintf(fp, "%u\n", this->getPolynomial());
    fprintf(fp, "%lu\n", this->getElements().size());

    for (Element *e : this->getElements()) {
        fprintf(fp, "%lu %lu %lu %lu %lu %lu\n", e->k, e->l, e->x1, e->y1, e->x2, e->y2);
    }

    fprintf(fp, "\n%lu\n", this->nodes.size());
    for (Node *n : this->nodes) {
        fprintf(fp, "%u %lu ", n->getId(), n->getElements().size());
        for (Element *e : n->getElements()) {
            fprintf(fp, "%lu %lu ", e->k, e->l);
        }
        if (n->getElements().size() > 1) {
            fprintf(fp, "%u %u ", n->getLeft()->getId(), n->getRight()->getId());
        }
        fprintf(fp, "%s\n", n->getProduction().c_str());
    }
    fclose(fp);
    return true;
}

Mesh *Mesh::loadFromFile(const char *filename, MeshSource src)
{
    FILE *fp;
    Mesh *mesh;
    fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("fopen");
        return NULL;
    }
    // read the P
    uint64_t p = 0;
    uint64_t elements = 0;
    uint64_t nodes = 0;

    fscanf(fp, "%lu", &p);
    fscanf(fp, "%lu", &elements);

    mesh = new Mesh(p, 2);

    std::map<std::tuple<uint64_t, uint64_t>, Element*> elementsMap;
    std::vector<Node *> nodesVector;

    for (uint64_t i=0; i<elements; ++i) {
        uint64_t k, l;
        uint64_t x1, y1, x2, y2;
        fscanf(fp, "%lu %lu %lu %lu %lu %lu", &k, &l, &x1, &y1, &x2, &y2);
        Element *e = new Element();
        e->x1 = x1;
        e->x2 = x2;
        e->y1 = y1;
        e->y2 = y2;
        e->k = k;
        e->l = l;
        std::tuple<uint64_t, uint64_t> t(k,l);
        mesh->addElement(e);
        elementsMap[t] = e;
    }

    fscanf(fp, "%lu", &nodes);

    for (uint64_t i=0; i<nodes; ++i) {
        uint64_t node_id;
        uint64_t nr_elems;
        fscanf(fp, "%lu %lu", &node_id, &nr_elems);
        Node *n = new Node(node_id);
        nodesVector.push_back(n);
        for (uint64_t q=0; q<nr_elems; ++q) {
            uint64_t k, l;
            fscanf(fp, "%lu %lu", &k, &l);
            n->addElement(elementsMap[std::tuple<uint64_t,uint64_t>(k,l)]);
        }
        if (nr_elems > 1) {
            uint64_t leftSon, rightSon;
            fscanf(fp, "%lu %lu", &leftSon, &rightSon);
            n->n_left = leftSon;
            n->n_right = rightSon;
        }
    }

    // all nodes read? built the Tree!

    for (uint64_t i=0; i<nodes; ++i) {
        if (nodesVector[i]->n_left != -1) {
            nodesVector[i]->setLeft(nodesVector[nodesVector[i]->n_left-1]);
        }
        if (nodesVector[i]->n_right != -1) {
            nodesVector[i]->setRight(nodesVector[nodesVector[i]->n_right-1]);
        }
        mesh->addNode(nodesVector[i]);
    }

    fclose(fp);
    return mesh;
}
