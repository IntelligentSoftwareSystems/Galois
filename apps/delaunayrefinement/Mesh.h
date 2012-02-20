/* 
 
 Lonestar DelaunayRefinement: Refinement of an initial, unrefined Delaunay
 mesh to eliminate triangles with angles < 30 degrees, using a
 variation of Chew's algorithm.
 
 Authors: Milind Kulkarni 
 
 Copyright (C) 2007, 2008 The University of Texas at Austin
 
 Licensed under the Eclipse Public License, Version 1.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.eclipse.org/legal/epl-v10.html
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 File: Mesh.h
 
 Modified: February 12th, 2008 by Milind Kulkarni (initial version)
 
 */ 

#ifndef _MESH_H_
#define _MESH_H_

#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <istream>

struct is_bad {
  Graph* mesh;
  is_bad(Graph* g) :mesh(g) {}
  bool operator()(const GNode& n) const {
    return mesh->getData(n, Galois::NONE).isBad();
  }
};

/**
 * Helper class used providing methods to read in information and create the graph 
 *
 */
class Mesh {
  // used during reading of the data and creation of the graph, to record edges between nodes of the graph 
  std::map<Edge, GNode> edge_map;

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

  void readElements(Graph* mesh, std::string filename, std::vector<Tuple>& tuples) {
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
      addElement(mesh, e);
    }
  }

  void readPoly(Graph* mesh, std::string filename, std::vector<Tuple>& tuples) {
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
      addElement(mesh, e);
    }
  }
  
  GNode addElement(Graph* mesh, Element& element) {
    GNode node = mesh->createNode(element);
    for (int i = 0; i < element.numEdges(); i++) {
      Edge edge = element.getEdge(i);
      if (edge_map.find(edge) == edge_map.end()) {
        edge_map[edge] = node;
      } else {
        mesh->addEdge(node, edge_map[edge], Galois::NONE);//, edge);
        edge_map.erase(edge);
      }
    }
    return node;
  }
  // .poly contains the perimeter of the mesh; edges basically, which is why it contains pairs of nodes
public:
  void read(Graph* mesh, std::string basename) {
    std::vector<Tuple> tuples;
    readNodes(basename, tuples);
    readElements(mesh, basename, tuples);
    readPoly(mesh, basename, tuples);
  }

  bool verify(Graph* mesh) {
    // ensure consistency of elements
    bool error = false;
     
    for (Graph::active_iterator ii = mesh->active_begin(), ee = mesh->active_end(); ii != ee; ++ii) {
 
      GNode node = *ii;
      Element& element = mesh->getData(node,Galois::NONE);
      int nsize = std::distance(mesh->edge_begin(node, Galois::NONE), mesh->edge_end(node, Galois::NONE));
      if (element.getDim() == 2) {
        if (nsize != 1) {
          std::cerr << "-> Segment " << element << " has " << nsize << " relation(s)\n";
          error = true;
        }
      } else if (element.getDim() == 3) {
        if (nsize != 3) {
          std::cerr << "-> Triangle " << element << " has " << nsize << " relation(s)";
          error = true;
        }
      } else {
        std::cerr << "-> Figures with " << element.getDim() << " edges";
        error = true;
      }
    }
    
    if (error)
      return false;
    
    // ensure reachability
    std::stack<GNode> remaining;
    std::set<GNode> found;
    remaining.push(*(mesh->active_begin()));
      
    while (!remaining.empty()) {
      GNode node = remaining.top();
      remaining.pop();
      if (!found.count(node)) {
        assert(mesh->containsNode(node) && "Reachable node was removed from graph");
        found.insert(node);
        int i = 0;
        for (Graph::edge_iterator ii = mesh->edge_begin(node, Galois::NONE), ee = mesh->edge_end(node, Galois::NONE); ii != ee; ++ii) {
          assert(i < 3);
          assert(mesh->containsNode(mesh->getEdgeDst(ii)));
          assert(node != mesh->getEdgeDst(ii));
          ++i;
          //          if (!found.count(*ii))
	  remaining.push(mesh->getEdgeDst(ii));
        }
      }
    }
    size_t msize = std::distance(mesh->active_begin(), mesh->active_end());

    if (found.size() != msize) {
      std::cerr << "Not all elements are reachable \n";
      std::cerr << "Found: " << found.size() << "\nMesh: " << msize << "\n";
      assert(0 && "Not all elements are reachable");
      return false;
    }
    return true;
  }
};

#endif
