/** Delaunay refinement -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * @author Milind Kulkarni <milind@purdue.edu>>
 */
#ifndef _MESH_H_
#define _MESH_H_

#include <vector>
#include <set>
#include <map>
#include <stack>
#include <cstdio>

struct is_bad {
  Graph* mesh;
  is_bad(Graph* g) :mesh(g) {}
  bool operator()(const GNode& n) const {
    return mesh->getData(n, Galois::NONE).isBad();
  }
};

struct processCreate {
  Graph* lmesh;
  processCreate(Graph* _lmesh) :lmesh(_lmesh) {}
  void operator()(Element& item) {
    GNode n = lmesh->createNode(item);
    lmesh->addNode(n);
  }
};

struct centerCmp {
  bool operator()(const Element& lhs, const Element& rhs) const {
    return lhs.getCenter() < rhs.getCenter();
  }
};

/**
 * Helper class used providing methods to read in information and create the graph 
 *
 */
class Mesh {
  std::vector<Element> elements;

private:
  void checkResults(int act, int exp, std::string& str) {
    if (act != exp) {
      std::cerr << "Failed read in " << str << "\n";
      abort();
    }
  }

  void readNodes(std::string filename, std::vector<Tuple>& tuples) {
    FILE* pFile = fopen(filename.append(".node").c_str(), "r");
    if (!pFile) {
      std::cerr << "Failed to load file " << filename << "\n";
      abort();
    }
    unsigned ntups;
    int r = fscanf(pFile, "%u %*u %*u %*u", &ntups);
    checkResults(r, 1, filename);
    tuples.resize(ntups);
    for (size_t i = 0; i < ntups; i++) {
      unsigned index;
      double x, y;
      r = fscanf(pFile, "%u %lf %lf %*f", &index, &x, &y);
      checkResults(r, 3, filename);
      tuples[index] = Tuple(x,y);
    }
    fclose(pFile);
  }

  void readElements(std::string filename, std::vector<Tuple>& tuples) {
    FILE* pFile = fopen(filename.append(".ele").c_str(), "r");
    if (!pFile) {
      std::cerr << "Failed to load file " << filename << "\n";
      abort();
    }
    unsigned nels;
    int r = fscanf(pFile, "%u %*u %*u", &nels);
    checkResults(r, 1, filename);
    for (size_t i = 0; i < nels; i++) {
      unsigned index;
      unsigned n1, n2, n3;
      r = fscanf(pFile, "%u %u %u %u", &index, &n1, &n2, &n3);
      checkResults(r, 4, filename);
      assert(n1 >= 0 && n1 < tuples.size());
      assert(n2 >= 0 && n2 < tuples.size());
      assert(n3 >= 0 && n3 < tuples.size());
      Element e(tuples[n1], tuples[n2], tuples[n3]);
      elements.push_back(e);
    }
    fclose(pFile);
  }

  void readPoly(std::string filename, std::vector<Tuple>& tuples) {
    FILE* pFile = fopen(filename.append(".poly").c_str(), "r");
    if (!pFile) {
      std::cerr << "Failed to load file " << filename << "\n";
      abort();
    }
    unsigned nsegs;
    int r = fscanf(pFile, "%*u %*u %*u %*u");
    checkResults(r, 0, filename);
    r = fscanf(pFile, "%u %*u", &nsegs);
    checkResults(r, 1, filename);
    for (size_t i = 0; i < nsegs; i++) {
      unsigned index, n1, n2;
      r = fscanf(pFile, "%u %u %u %*u", &index, &n1, &n2);
      checkResults(r, 3, filename);
      assert(n1 >= 0 && n1 < tuples.size());
      assert(n2 >= 0 && n2 < tuples.size());
      Element e(tuples[n1], tuples[n2]);
      elements.push_back(e);
    }
    fclose(pFile);
  }
  
  void addElement(Graph* mesh, GNode node, std::map<Edge, GNode>& edge_map) {
    Element& element = mesh->getData(node);
    for (int i = 0; i < element.numEdges(); i++) {
      Edge edge = element.getEdge(i);
      if (edge_map.find(edge) == edge_map.end()) {
        edge_map[edge] = node;
      } else {
        mesh->addEdge(node, edge_map[edge], Galois::NONE);//, edge);
        edge_map.erase(edge);
      }
    }
  }

  void makeGraph(Graph* mesh) {
    std::sort(elements.begin(), elements.end(), centerCmp());
    Galois::do_all(elements.begin(), elements.end(), processCreate(mesh));
    std::map<Edge, GNode> edge_map;
    for (Graph::iterator ii = mesh->begin(), ee = mesh->end();
	 ii != ee; ++ii)
      addElement(mesh, *ii, edge_map);
  }

  // .poly contains the perimeter of the mesh; edges basically, which is why it contains pairs of nodes
public:
  void read(Graph* mesh, std::string basename) {
    std::vector<Tuple> tuples;
    readNodes(basename, tuples);
    readElements(basename, tuples);
    readPoly(basename, tuples);
    makeGraph(mesh);
  }
};

#endif
