#ifndef ANALYSIS_HPP
#define ANALYSIS_HPP
#include <set>
#include <map>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include "Mesh.hpp"

typedef std::tuple<uint64_t, uint64_t> vertex;
typedef std::tuple<vertex, vertex> edge;
typedef std::tuple<vertex, vertex> face;

class Analysis {
    private:
        static void mergeAnaliser(Node *node);
        
    public:
        static void doAnalise(Mesh *mesh);
        static void nodeAnaliser(Node *n, std::set<uint64_t> *parent);

        // for debug use
        static void debugNode(Node *n);
        static void printTree(Node *n);
        static void printElement(Element *e);

};

#endif // ANALYSIS_HPP
