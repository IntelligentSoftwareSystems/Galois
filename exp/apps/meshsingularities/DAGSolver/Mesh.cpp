#include "Mesh.hpp"
#include <set>
#include <map>
#include <tuple>
#include <algorithm>

void Mesh::addElement(Element *e)
{
    this->elements.push_back(e);
}

void Mesh::addNode(Node *n)
{
    this->nodes.push_back(n);
}

void Mesh::addAdaptations(MeshSingularity sing)
{
    int max_level = 0;
    std::vector<Element*> edgeElements;

    for (Element *e : this->getElements()) {
        e->x1 = 2*e->x1;
        e->x2 = 2*e->x2;
        e->y1 = 2*e->y1;
        e->y2 = 2*e->y2;
        max_level = e->k > max_level ? e->k : max_level;

        if (e->y1 == 0) {
            edgeElements.push_back(e);
        }
    }

    if (sing == EDGE) {
        int l = 1;
        for (Element *e : edgeElements) {
            // here we need to break elements into smaller ones,
            // let's create four elements:
            Element *e1 = new Element();
            Element *e2 = new Element();
            Element *e3 = new Element();
            Element *e4 = new Element();

            e1->x1 = e->x1;
            e1->x2 = (e->x1+e->x2)/2;
            e1->y1 = e->y1;
            e1->y1 = (e->y1+e->y2)/2;
            e1->k = e->k+1;
            e1->l = l++;

            e2->x1 = (e->x1+e->x2)/2;
            e2->x2 = e->x2;
            e2->y1 = e->y1;
            e2->y2 = (e->y1+e->y2)/2;
            e2->k = e->k+1;
            e2->l = l++;

            e3->x1 = e->x1;
            e3->x2 = (e->x1+e->x2)/2;
            e3->y1 = (e->y1+e->y2)/2;
            e3->y2 = e->y2;
            e3->k = e->k+1;
            e3->l = l++;

            e4->x1 = (e->x1+e->x2)/2;
            e4->x2 = e->x2;
            e4->y1 = (e->y1+e->y2)/2;
            e4->y2 = e->y2;
            e4->k = e->k+1;
            e4->l = l++;

            std::vector<Element*>::iterator it = this->getElements().begin();
            for (; it!=this->getElements().end(); ++it) {
                if (*it == e) {
                    this->getElements().erase(it);
                    break;
                }
            }

            this->addElement(e1);
            this->addElement(e2);
            this->addElement(e3);
            this->addElement(e4);



        }
    }
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
    
    auto myCompFunction = [] (Node * n1, Node * n2){
        return (n1->getId() < n2->getId());
    };
    std::sort(this->nodes.begin(), this->nodes.end(), myCompFunction);

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
    nodesVector.resize(nodes);

    for (uint64_t i=0; i<nodes; ++i) {
        uint64_t node_id;
        uint64_t nr_elems;
        fscanf(fp, "%lu %lu", &node_id, &nr_elems);
        Node *n = new Node(node_id);
        nodesVector[node_id-1] = n;
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
            nodesVector[nodesVector[i]->n_left-1]->setParent(nodesVector[i]);
        }
        if (nodesVector[i]->n_right != -1) {
            nodesVector[i]->setRight(nodesVector[nodesVector[i]->n_right-1]);
            nodesVector[nodesVector[i]->n_right-1]->setParent(nodesVector[i]);
        }
        mesh->addNode(nodesVector[i]);
    }

    fclose(fp);
    return mesh;
}

bool Mesh::loadFrontalMatrices(const char *filename)
{
    std::map<std::tuple<int,int>, EquationSystem*> inputMatrices;
    std::map<int,std::tuple<int,int>> levelMaps;

    for (Element *e : this->elements) {
        std::tuple<int, int> t(e->k, e->l);
        inputMatrices[t] = new EquationSystem(e->dofs.size());
    }

    FILE *fp = NULL;

    if ((fp=fopen(filename, "r")) == NULL) {
        perror("fopen");
        return false;
    }

    for (int i=0; i<this->elements.size(); ++i) {
        int k, l, level;
        fscanf(fp, "%d %d %d", &k, &l, &level);

        std::tuple<int, int> t(k,l);
        levelMaps[level] = t;
    }

    for (int i=0; i<this->elements.size(); ++i) {
        int level;
        fscanf(fp, "%d", &level);
        EquationSystem *e = inputMatrices[levelMaps[level]];

        for (int j=0; j<e->n; ++j) {
            double val;
            fscanf(fp, "%g ", &val);
            e->rhs[j];
        }

        for (int j=0; j<e->n; ++j) {
            for (int k=0; k<=j; ++k) {
                double val;
                fscanf(fp, "%g ", &val);
                e->matrix[j][k] = e->matrix[k][j] = val;
            }
        }
    }



    fclose(fp);
    return true;
}

void Mesh::setRootNode(Node* root)
{
    Node * oldRoot = this->nodes[0];
    int newRoot=0;
    bool found=false;
    while (!found){
        if (this->nodes[newRoot] == root){
            found = true;
        } else{
            newRoot++;
        }
    }
    this->nodes[newRoot] = oldRoot;
    this->nodes[0] = root;
}
