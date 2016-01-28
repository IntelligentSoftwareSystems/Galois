#ifndef MESH_HPP
#define MESH_HPP

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <tuple>
#include <map>
#include <string>
#include <vector>
#include "Element.hpp"
#include "Node.hpp"
#include <cmath>

class Mesh {
    private:
        std::vector<Node *> nodes;
	std::vector<uint64_t> supernodes;
        std::vector<Element *> elements;
        uint64_t supernode_count = 0;
        Node * root = NULL;

    public:
        Mesh() {}
        void addNode(Node *n);
        void addElement(Element *e);
        uint64_t getTotalDofs();
        static Mesh *loadFromFile(const char *filename);
        bool loadFrontalMatrices(const char *filename);
        bool saveToFile(const char * filename);
        Node *getRootNode();
        std::vector<Element *> &getElements();
        void setSupernodes(uint64_t supernodes);
        uint64_t getSupernodes();
        
        void setRootNode(Node * root);

        ~Mesh() {
            for (Node *n : nodes) {
                delete (n);
            }
            for (Element *e : elements) {
                delete (e);
            }
        }

};

#endif // MESH_HPP
