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
#include <fstream>
#include <istream>

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
  template<typename Context>
  void operator()(Element item, Context& lwl) {
    lmesh->createNode(item);
  }
};

/**
 * Helper class used providing methods to read in information and create the graph 
 *
 */
class Mesh {
  std::vector<Element> elements;

private:
  void next_line(std::ifstream& scanner) {
    scanner.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  void readNodes(std::string filename, std::vector<Tuple>& tuples) {
    std::ifstream scanner(filename.append(".node").c_str());
    size_t ntups;
    scanner >> ntups;
    next_line(scanner);

    tuples.resize(ntups);
    for (size_t i = 0; i < ntups; i++) {
      size_t index;
      double x;
      double y;
      scanner >> index >> x >> y;
      next_line(scanner);
      tuples[index] = Tuple(x, y);
    }
  }

  void readElements(std::string filename, std::vector<Tuple>& tuples) {
    std::ifstream scanner(filename.append(".ele").c_str());
    
    size_t nels;
    scanner >> nels;
    next_line(scanner);

    for (size_t i = 0; i < nels; i++) {
      size_t index;
      size_t n1, n2, n3;
      scanner >> index >> n1 >> n2 >> n3;
      assert(n1 >= 0 && n1 < tuples.size());
      assert(n2 >= 0 && n2 < tuples.size());
      assert(n3 >= 0 && n3 < tuples.size());
      Element e(tuples[n1], tuples[n2], tuples[n3]);
      elements.push_back(e);
    }
  }

  void readPoly(std::string filename, std::vector<Tuple>& tuples) {
    std::ifstream scanner(filename.append(".poly").c_str());
    next_line(scanner);
    size_t nsegs;
    scanner >> nsegs;
    next_line(scanner);
    for (size_t i = 0; i < nsegs; i++) {
      size_t index;
      size_t n1;
      size_t n2;
      scanner >> index >> n1 >> n2;
      assert(n1 >= 0 && n1 < tuples.size());
      assert(n2 >= 0 && n2 < tuples.size());
      next_line(scanner);
      Element e(tuples[n1], tuples[n2]);
      elements.push_back(e);
    }
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
    Galois::for_each<>(elements.begin(), elements.end(), processCreate(mesh));
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
