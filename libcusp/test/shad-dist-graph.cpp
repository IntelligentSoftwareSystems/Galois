/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include <fstream>

#include "galois/Galois.h"
#include "galois/graphs/CuSPPartitioner.h"
#include "shad/ShadGraphConverter.h"

int main() {
  galois::DistMemSys G;
  unsigned M = galois::substrate::getThreadPool().getMaxThreads();
  // M = 1;
  galois::setActiveThreads(M);

  shad::ShadGraphConverter shadConverter;
  size_t numNodes{0}, numEdges{0};

  // TODO(hc): This path should be properly set based on user's environment.
  // Later, this test dataset will be included in the Galois repository, and
  // will use a relative path.
  std::string filename = "/home/hochan/data.01.csv";
  shadConverter.readSHADFile(filename, &numNodes, &numEdges);
  std::unique_ptr<galois::graphs::DistGraph<shad::ShadNodeTy, shad::ShadEdgeTy>>
      graph = galois::cuspPartitionGraph<GenericCVC, shad::ShadNodeTy,
                                         shad::ShadEdgeTy>(
          filename, galois::CUSP_CSR, galois::CUSP_CSR, true, true);

  std::cout << "Test starts...\n";

  galois::DGAccumulator<uint64_t> sumGlobalNodes;
  galois::DGAccumulator<uint64_t> sumGlobalEdges;

  sumGlobalNodes.reset();
  sumGlobalEdges.reset();

  sumGlobalNodes += graph->numMasters();
  sumGlobalEdges += graph->sizeEdges();

  uint64_t reducedSumGlobalNodes = sumGlobalNodes.reduce();
  uint64_t reducedSumGlobalEdges = sumGlobalEdges.reduce();

  assert(reducedSumGlobalNodes == numNodes);
  assert(reducedSumGlobalNodes == graph->globalSize());
  assert(reducedSumGlobalEdges == numEdges);
  assert(reducedSumGlobalEdges == graph->globalSizeEdges());

  std::cout << "Num. nodes/edges tests has been passed\n";

  uint32_t id       = galois::runtime::getSystemNetworkInterface().ID;
  uint32_t numHosts = galois::runtime::getSystemNetworkInterface().Num;
  {
    std::ofstream fp(std::to_string(id) + ".master");
    for (uint32_t src = 0; src < graph->numMasters(); ++src) {
      uint64_t srcglobal = graph->getGID(src);
      fp << "node " << srcglobal << ", type: " << graph->getData(src).type
         << ", key: " << graph->getData(src).key << "\n";
      for (auto e : graph->edges(src)) {
        uint32_t dstlocal  = graph->getEdgeDst(e);
        uint64_t dstglobal = graph->getGID(dstlocal);
        fp << "\t edge dst " << dstglobal << ", type: " << graph->getEdgeData(e)
           << "\n";
      }
    }
    fp.close();
  }

  {
    for (uint32_t host = 0; host < numHosts; ++host) {
      if (host == id) {
        continue;
      }
      std::ofstream fp(std::to_string(id) + "-" + std::to_string(host) +
                       ".graph");
      for (uint32_t i = 0; i < graph->size(); ++i) {
        fp << i << ", " << graph->getGID(i) << ", " << graph->getData(i).type
           << ", " << graph->getData(i).key << "\n";
      }
      fp.close();
    }
  }
#if 0
  {
  for (uint32_t host = 0; host < numHosts; ++host) {
    if (host == id) {
      continue;
    }
    std::ofstream fp(std::to_string(id) + "-" + std::to_string(host) + ".mirror");
    for (uint32_t i = 0;
         i < graph->getMirrorNodes()[host].size(); ++i) {
      uint64_t srcglobal = graph->getMirrorNodes()[host][i];
      uint32_t src = graph->getLID(srcglobal);
      fp << "src:" << src << ", global:" << srcglobal << ", node data:" <<
        graph->getData(src) << "\n" << std::flush;

      assert(shadConverter.checkNode(srcglobal, graph->getData(src)));
      fp << "node " << srcglobal << ", type: " << graph->getData(src) << "\n";
      //if (std::distance(graph->edge_begin(src), graph->edge_end(src)) > 0) {
        for (auto e : graph->edges(src)) {
          uint32_t dst = graph->getEdgeDst(e);
          uint64_t dstglobal = graph->getGID(dst);
          assert(shadConverter.checkNode(dstglobal, graph->getData(dst)));
          assert(shadConverter.checkEdge(srcglobal, dstglobal,
              std::distance(graph->edge_begin(src), e),
              graph->getEdgeData(e)));
          fp << "\t edge dst " << dstglobal << ", type: " <<
              graph->getEdgeData(e) << "\n" << std::flush;
        }
    }
    fp.close();
    }
  }
#endif

  return 0;
}
