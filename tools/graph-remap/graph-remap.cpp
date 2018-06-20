/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/BufferedGraph.h"
#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> mappingFilename(cll::Positional,
                                             cll::desc("<mapping file>"),
                                             cll::Required);
static cll::opt<std::string>
    outputFilename(cll::Positional, cll::desc("<output file>"), cll::Required);

using Writer = galois::graphs::FileGraphWriter;

/**
 * Create node map from file
 */
std::map<uint32_t, uint32_t> createNodeMap() {
  galois::gInfo("Creating node map");
  // read new mapping
  std::ifstream mapFile;
  mapFile.open(mappingFilename);
  int64_t endOfFile = mapFile.seekg(0, std::ios_base::end).tellg();
  mapFile.seekg(0, std::ios_base::beg);

  // remap node listed on line n in the mapping to node n
  std::map<uint32_t, uint32_t> remapper;
  uint64_t counter = 0;
  while (((int64_t)mapFile.tellg() + 1) != endOfFile) {
    uint64_t nodeID;
    mapFile >> nodeID;
    remapper[nodeID] = counter++;
  }

  GALOIS_ASSERT(remapper.size() == counter);
  galois::gInfo("Remapping ", counter, " nodes");
  mapFile.close();

  galois::gInfo("Node map created");

  return remapper;
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::map<uint32_t, uint32_t> remapper = createNodeMap();

  galois::gInfo("Loading graph to remap");
  galois::graphs::BufferedGraph<void> graphToRemap;
  graphToRemap.loadGraph(inputFilename);
  galois::gInfo("Graph loaded");

  Writer graphWriter;
  graphWriter.setNumNodes(remapper.size());
  graphWriter.setNumEdges(graphToRemap.sizeEdges());

  // phase 1: count degrees
  graphWriter.phase1();
  galois::gInfo("Starting degree counting");
  size_t prevNumNodes  = graphToRemap.size();
  size_t nodeIDCounter = 0;
  for (size_t i = 0; i < prevNumNodes; i++) {
    // see if current node is to be remapped, i.e. exists in the map
    if (remapper.find(i) != remapper.end()) {
      GALOIS_ASSERT(nodeIDCounter == remapper[i]);
      for (auto e = graphToRemap.edgeBegin(i); e < graphToRemap.edgeEnd(i);
           e++) {
        graphWriter.incrementDegree(nodeIDCounter);
      }
      nodeIDCounter++;
    }
  }
  GALOIS_ASSERT(nodeIDCounter == remapper.size());

  // phase 2: edge construction
  graphWriter.phase2();
  galois::gInfo("Starting edge construction");
  nodeIDCounter = 0;
  for (size_t i = 0; i < prevNumNodes; i++) {
    // see if current node is to be remapped, i.e. exists in the map
    if (remapper.find(i) != remapper.end()) {
      GALOIS_ASSERT(nodeIDCounter == remapper[i]);
      for (auto e = graphToRemap.edgeBegin(i); e < graphToRemap.edgeEnd(i);
           e++) {
        uint32_t dst = graphToRemap.edgeDestination(*e);
        GALOIS_ASSERT(remapper.find(dst) != remapper.end());
        graphWriter.addNeighbor(nodeIDCounter, remapper[dst]);
      }
      nodeIDCounter++;
    }
  }
  GALOIS_ASSERT(nodeIDCounter == remapper.size());

  galois::gInfo("Finishing up: outputting graph shortly");

  graphWriter.finish<void>();
  graphWriter.toFile(outputFilename);

  galois::gInfo("new size is ", graphWriter.size(), " num edges ",
                graphWriter.sizeEdges());

  return 0;
}
