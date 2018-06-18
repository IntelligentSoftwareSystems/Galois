#include "Mesh.hpp"
#include <set>
#include <map>
#include <tuple>
#include <algorithm>

void Mesh::addNode(Node* n) { this->nodes.push_back(n); }

void Mesh::addElement(Element* e) { this->elements.push_back(e); }

void Mesh::setSupernodes(uint64_t supernodes) {
  this->supernode_count = supernodes;
  this->supernodes.resize(supernodes);
}

uint64_t Mesh::getSupernodes() { return this->supernode_count; }

uint64_t Mesh::getTotalDofs() {
  uint64_t total;

  total = 0;
  for (uint64_t i = 0; i < this->supernodes.size(); ++i)
    total += this->supernodes[i];
  return total;
}

std::vector<Element*>& Mesh::getElements() { return this->elements; }

Node* Mesh::getRootNode() { return this->root; }

bool Mesh::saveToFile(const char* filename) {
  FILE* fp;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    return false;
  }

  auto myCompFunction = [](Node* n1, Node* n2) {
    return (n1->getId() < n2->getId());
  };
  std::sort(this->nodes.begin(), this->nodes.end(), myCompFunction);

  fprintf(fp, "%lu\n", this->supernodes.size());

  for (uint64_t i = 0; i < this->supernodes.size(); ++i) {
    fprintf(fp, "%lu %lu\n", i, supernodes[i]);
  }

  fprintf(fp, "%lu\n", this->getElements().size());

  for (Element* e : this->getElements()) {
    fprintf(fp, "%lu %lu\n", e->k, e->l);
  }

  fprintf(fp, "\n%lu\n", this->nodes.size());
  for (Node* n : this->nodes) {
    fprintf(fp, "%u %lu ", n->getId(), n->getElements().size());
    for (Element* e : n->getElements()) {
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

Mesh* Mesh::loadFromFile(const char* filename) {
  FILE* fp;
  Mesh* mesh;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    perror("fopen");
    return NULL;
  }
  uint64_t supernodes = 0;
  uint64_t nodes      = 0;
  uint64_t elements;

  fscanf(fp, "%lu", &supernodes);

  mesh = new Mesh();
  mesh->setSupernodes(supernodes);
  for (uint64_t i = 0; i < supernodes; ++i) {
    uint64_t tmp;
    uint64_t id;
    fscanf(fp, "%lu %lu", &id, &tmp);
    mesh->supernodes[i] = tmp;
  }

  fscanf(fp, "%lu", &elements);

  std::map<std::tuple<uint64_t, uint64_t>, Element*> elementsMap;
  std::vector<Node*> nodesVector;

  for (uint64_t i = 0; i < elements; ++i) {
    uint64_t k, l;
    fscanf(fp, "%lu %lu", &k, &l);
    Element* e = new Element();
    e->k       = k;
    e->l       = l;
    std::tuple<uint64_t, uint64_t> t(k, l);
    mesh->addElement(e);
    elementsMap[t] = e;
    fscanf(fp, "%lu", &k);
    for (uint64_t j = 0; j < k; ++j) {
      fscanf(fp, "%lu", &l);
      e->supernodes.push_back(l - 1);
    }
  }

  fscanf(fp, "%lu", &nodes);
  nodesVector.resize(nodes);

  for (uint64_t i = 0; i < nodes; ++i) {
    uint64_t node_id;
    uint64_t nr_elems;
    fscanf(fp, "%lu %lu", &node_id, &nr_elems);
    Node* n                  = new Node(&mesh->supernodes, node_id);
    nodesVector[node_id - 1] = n;
    for (uint64_t q = 0; q < nr_elems; ++q) {
      uint64_t k, l;
      fscanf(fp, "%lu %lu", &k, &l);
      n->addElement(elementsMap[std::tuple<uint64_t, uint64_t>(k, l)]);
    }
    if (nr_elems > 1) {
      uint64_t leftSon, rightSon;
      fscanf(fp, "%lu %lu", &leftSon, &rightSon);
      n->n_left  = leftSon;
      n->n_right = rightSon;
    } else {
      n->n_left = n->n_right = -1;
    }
    if (i == 0) {
      mesh->root = n;
    }
  }

  // all nodes read? built the Tree!

  for (uint64_t i = 0; i < nodes; ++i) {
    if (nodesVector[i]->n_left != -1) {
      nodesVector[i]->setLeft(nodesVector[nodesVector[i]->n_left - 1]);
      nodesVector[nodesVector[i]->n_left - 1]->setParent(nodesVector[i]);
    }
    if (nodesVector[i]->n_right != -1) {
      nodesVector[i]->setRight(nodesVector[nodesVector[i]->n_right - 1]);
      nodesVector[nodesVector[i]->n_right - 1]->setParent(nodesVector[i]);
    }
    mesh->addNode(nodesVector[i]);
  }

  fclose(fp);
  return mesh;
}

bool Mesh::loadFrontalMatrices(const char* filename) {
  std::map<std::tuple<int, int>, EquationSystem*> inputMatrices;
  std::map<int, std::tuple<int, int>> levelMaps;

  for (Element* e : this->elements) {
    std::tuple<int, int> t(e->k, e->l);
    inputMatrices[t] = new EquationSystem(e->supernodes.size());
  }

  FILE* fp = NULL;

  if ((fp = fopen(filename, "r")) == NULL) {
    perror("fopen");
    return false;
  }

  for (int i = 0; i < this->elements.size(); ++i) {
    int k, l, level;
    fscanf(fp, "%d %d %d", &k, &l, &level);

    std::tuple<int, int> t(k, l);
    levelMaps[level] = t;
  }

  for (int i = 0; i < this->elements.size(); ++i) {
    int level;
    fscanf(fp, "%d", &level);
    EquationSystem* e = inputMatrices[levelMaps[level]];

    for (int j = 0; j < e->n; ++j) {
      double val;
      fscanf(fp, "%lg ", &val);
    }

    for (int j = 0; j < e->n; ++j) {
      for (int k = 0; k <= j; ++k) {
        double val;
        fscanf(fp, "%lg ", &val);
        e->matrix[j][k] = e->matrix[k][j] = val;
      }
    }
  }

  fclose(fp);
  return true;
}

void Mesh::setRootNode(Node* root) {
  Node* oldRoot = this->nodes[0];
  int newRoot   = 0;
  bool found    = false;
  while (!found) {
    if (this->nodes[newRoot] == root) {
      found = true;
    } else {
      newRoot++;
    }
  }
  this->nodes[newRoot] = oldRoot;
  this->nodes[0]       = root;
}
