/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef MESH_H
#define MESH_H

#include "Subgraph.h"

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <cstdio>


struct is_bad {
  Graph& g;
  is_bad(Graph& _g): g(_g) {}
  bool operator()(const GNode& n) const {
    return g.getData(n, galois::MethodFlag::UNPROTECTED).isBad();
  }
};

struct centerXCmp {
  bool operator()(const Element& lhs, const Element& rhs) const {
    //return lhs.getCenter() < rhs.getCenter();
    return lhs.getPoint(0)[0] < rhs.getPoint(0)[0];
  }
};

struct centerYCmp {
  bool operator()(const Element& lhs, const Element& rhs) const {
    //return lhs.getCenter() < rhs.getCenter();
    return lhs.getPoint(0)[1] < rhs.getPoint(0)[1];
  }
};

struct centerYCmpInv {
  bool operator()(const Element& lhs, const Element& rhs) const {
    //return lhs.getCenter() < rhs.getCenter();
    return rhs.getPoint(0)[1] < lhs.getPoint(0)[1];
  }
};

/**
 * Helper class used providing methods to read in information and create the graph 
 *
 */
class Mesh {
  std::vector<Element> elements;
  size_t id;

private:
  void checkResults(int act, int exp, std::string& str) {
    if (act != exp) {
      std::cerr << "Failed read in " << str << "\n";
      abort();
    }
  }

  bool readNodesBin(std::string filename, std::vector<Tuple>& tuples) {
    FILE* pFile = fopen(filename.append(".node.bin").c_str(), "r");
    if (!pFile) {
      return false;
    }
    std::cout << "Using bin for node\n";
    uint32_t ntups[4];
    if (fread(&ntups[0], sizeof(uint32_t), 4, pFile) < 4) {
      std::cerr << "Malformed binary file\n";
      abort();
    }
    tuples.resize(ntups[0]);
    for (size_t i = 0; i < ntups[0]; i++) {
      struct record {
	uint32_t index;
	double x, y, z;
      };
      record R;
      if (fread(&R, sizeof(record), 1, pFile) < 1) {
        std::cerr << "Malformed binary file\n";
        abort();
      }
      tuples[R.index] = Tuple(R.x,R.y);
    }
    fclose(pFile);
    return true;
  }

  void readNodes(std::string filename, std::vector<Tuple>& tuples) {
    if (readNodesBin(filename, tuples))
      return;
    else
      writeNodes(filename);
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

  void writeNodes(std::string filename) {
    std::string filename2 = filename;
    FILE* pFile = fopen(filename.append(".node").c_str(), "r");
    FILE* oFile = fopen(filename2.append(".node.bin").c_str(), "w");
    if (!pFile) {
      std::cerr << "Failed to load file " << filename << " (continuing)\n";
      return;
    }
    if (!oFile) {
      std::cerr << "Failed to open file " << filename2 << " (continuing)\n";
      return;
    }
    unsigned ntups[4];
    int r = fscanf(pFile, "%u %u %u %u", &ntups[0], &ntups[1], &ntups[2], &ntups[3]);
    checkResults(r, 4, filename);
    uint32_t ntups32[4] = {ntups[0], ntups[1], ntups[2], ntups[3]};
    fwrite(&ntups32[0], sizeof(uint32_t), 4, oFile);

    for (size_t i = 0; i < ntups[0]; i++) {
      struct record {
	unsigned index;
	double x, y, z;
      };
      struct recordOut {
	uint32_t index;
	double x, y, z;
      };
      record R;
      r = fscanf(pFile, "%u %lf %lf %lf", &R.index, &R.x, &R.y, &R.z);
      checkResults(r, 4, filename);
      recordOut R2 = {R.index, R.x, R.y, R.z};
      fwrite(&R2, sizeof(recordOut), 1, oFile);
    }
    fclose(pFile);
    fclose(oFile);
  }

  bool readElementsBin(std::string filename, std::vector<Tuple>& tuples) {
    FILE* pFile = fopen(filename.append(".ele.bin").c_str(), "r");
    if (!pFile) {
      return false;
    }
    std::cout << "Using bin for ele\n";
    uint32_t nels[3];
    if (fread(&nels[0], sizeof(uint32_t), 3, pFile) < 3) {
      std::cerr << "Malformed binary file\n";
      abort();
    }
    for (size_t i = 0; i < nels[0]; i++) {
      uint32_t r[4];
      if (fread(&r[0], sizeof(uint32_t), 4, pFile) < 4) {
        std::cerr << "Malformed binary file\n";
        abort();
      }
      assert(r[1] < tuples.size());
      assert(r[2] < tuples.size());
      assert(r[3] < tuples.size());
      Element e(tuples[r[1]], tuples[r[2]], tuples[r[3]], ++id);
      elements.push_back(e);
    }
    fclose(pFile);
    return true;
  }

  void readElements(std::string filename, std::vector<Tuple>& tuples) {
    if (readElementsBin(filename, tuples))
      return;
    else
      writeElements(filename);
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
      assert(n1 < tuples.size());
      assert(n2 < tuples.size());
      assert(n3 < tuples.size());
      Element e(tuples[n1], tuples[n2], tuples[n3], ++id);
      elements.push_back(e);
    }
    fclose(pFile);
  }

  void writeElements(std::string filename) {
    std::string filename2 = filename;
    FILE* pFile = fopen(filename.append(".ele").c_str(), "r");
    FILE* oFile = fopen(filename2.append(".ele.bin").c_str(), "w");
    if (!pFile) {
      std::cerr << "Failed to load file " << filename << " (continuing)\n";
      return;
    }
    if (!oFile) {
      std::cerr << "Failed to open file " << filename2 << " (continuing)\n";
      return;
    }
    unsigned nels[3];
    int r = fscanf(pFile, "%u %u %u", &nels[0], &nels[1], &nels[2]);
    checkResults(r, 3, filename);
    uint32_t nels32[3] = {nels[0], nels[1], nels[2]};
    fwrite(&nels32[0], sizeof(uint32_t), 3, oFile);

    for (size_t i = 0; i < nels[0]; i++) {
      unsigned index;
      unsigned n1, n2, n3;
      r = fscanf(pFile, "%u %u %u %u", &index, &n1, &n2, &n3);
      checkResults(r, 4, filename);
      uint32_t vals[4] = {index, n1, n2, n3};
      fwrite(&vals[0], sizeof(uint32_t), 4, oFile);
    }
    fclose(pFile);
    fclose(oFile);
  }

  bool readPolyBin(std::string filename, std::vector<Tuple>& tuples) {
    FILE* pFile = fopen(filename.append(".poly.bin").c_str(), "r");
    if (!pFile) {
      return false;
    }
    std::cout << "Using bin for poly\n";
    uint32_t nsegs[4];
    if (fread(&nsegs[0], sizeof(uint32_t), 4, pFile) < 4) {
      std::cerr << "Malformed binary file\n";
      abort();
    }
    if (fread(&nsegs[0], sizeof(uint32_t), 2, pFile) < 2) {
      std::cerr << "Malformed binary file\n";
      abort();
    }
    for (size_t i = 0; i < nsegs[0]; i++) {
      uint32_t r[4];
      if (fread(&r[0], sizeof(uint32_t), 4, pFile) < 4) {
        std::cerr << "Malformed binary file\n";
        abort();
      }
      assert(r[1] < tuples.size());
      assert(r[2] < tuples.size());
      Element e(tuples[r[1]], tuples[r[2]], ++id);
      elements.push_back(e);
    }
    fclose(pFile);
    return true;
  }

  void readPoly(std::string filename, std::vector<Tuple>& tuples) {
    if (readPolyBin(filename, tuples))
      return;
    else
      writePoly(filename);
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
      assert(n1 < tuples.size());
      assert(n2 < tuples.size());
      Element e(tuples[n1], tuples[n2], ++id);
      elements.push_back(e);
    }
    fclose(pFile);
  }

  void writePoly(std::string filename) {
    std::string filename2 = filename;
    FILE* pFile = fopen(filename.append(".poly").c_str(), "r");
    FILE* oFile = fopen(filename2.append(".poly.bin").c_str(), "w");
    if (!pFile) {
      std::cerr << "Failed to load file " << filename << " (continuing)\n";
      return;
    }
    if (!oFile) {
      std::cerr << "Failed to open file " << filename2 << " (continuing)\n";
      return;
    }
    unsigned nsegs[4];
    int r = fscanf(pFile, "%u %u %u %u", &nsegs[0], &nsegs[1], &nsegs[2], &nsegs[3]);
    checkResults(r, 4, filename);
    uint32_t nsegs32[4] = {nsegs[0], nsegs[1], nsegs[2], nsegs[3]};
    fwrite(&nsegs32[0], sizeof(uint32_t), 4, oFile);
    r = fscanf(pFile, "%u %u", &nsegs[0], &nsegs[1]);
    checkResults(r, 2, filename);
    nsegs32[0] = nsegs[0]; nsegs32[1] = nsegs[1];
    fwrite(&nsegs32[0], sizeof(uint32_t), 2, oFile);
    for (size_t i = 0; i < nsegs[0]; i++) {
      unsigned index, n1, n2, n3;
      r = fscanf(pFile, "%u %u %u %u", &index, &n1, &n2, &n3);
      checkResults(r, 4, filename);
      uint32_t r[4] = {index, n1, n2, n3};
      fwrite(&r[0], sizeof(uint32_t), 4, oFile);
    }
    fclose(pFile);
    fclose(oFile);
  }
  
  void addElement(Graph& mesh, GNode node, std::map<Edge, GNode>& edge_map) {
    Element& element = mesh.getData(node);
    for (int i = 0; i < element.numEdges(); i++) {
      Edge edge = element.getEdge(i);
      if (edge_map.find(edge) == edge_map.end()) {
        edge_map[edge] = node;
      } else {
        mesh.addEdge(node, edge_map[edge], galois::MethodFlag::UNPROTECTED);
        edge_map.erase(edge);
      }
    }
  }

  template<typename Iter>
  void divide(const Iter& b, const Iter& e) {
    if (std::distance(b,e) > 16) {
      std::sort(b,e, centerXCmp());
      Iter m = galois::split_range(b,e);
      std::sort(b,m, centerYCmpInv());
      std::sort(m,e, centerYCmp());
      divide(b, galois::split_range(b,m));
      divide(galois::split_range(b,m), m);
      divide(m,galois::split_range(m,e));
      divide(galois::split_range(m,e), e);
    }
  }

  template <typename L>
  void createNodes(Graph& g, const L& loop) {

    loop(galois::iterate(elements), 
        [&] (const Element& item) {
          GNode n = g.createNode(item);
          g.addNode(n);
        }
        , galois::loopname("allocate"));

  }
  void makeGraph(Graph& mesh, bool parallelAllocate) {
    //std::sort(elements.begin(), elements.end(), centerXCmp());
    divide(elements.begin(), elements.end());

    if (parallelAllocate) 
      createNodes(mesh, galois::DoAll());
    else
      createNodes(mesh, galois::StdForEach());

    std::map<Edge, GNode> edge_map;
    for (auto ii = mesh.begin(), ee = mesh.end(); ii != ee; ++ii)
      addElement(mesh, *ii, edge_map);
  }

public:
  Mesh(): id(0) { }

  void read(Graph& mesh, std::string basename, bool parallelAllocate) {
    std::vector<Tuple> tuples;
    readNodes(basename, tuples);
    readElements(basename, tuples);
    readPoly(basename, tuples);
    makeGraph(mesh, parallelAllocate);
  }
};

#endif
