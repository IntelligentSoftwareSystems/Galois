/*
 * Run this script in Debug view and compare its result to result of
 * MiningPartitioner.h
 *
 * A testing result sheet could be found at
 * https://docs.google.com/spreadsheets/u/1/d/1D0dAab29uazRKVroBZdEvAsEUT92aNHsiIYuHK6dlDA
 *
 * TODO: include a script to gen dataset and compare result
 *
 */
#include "galois/wmd/graph.h"
#include "galois/wmd/WMDPartitioner.h"

#include "galois/DistGalois.h"
#include "galois/graphs/GenericPartitioners.h"

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace agile::workflow1;

typedef galois::graphs::WMDGraph<agile::workflow1::Vertex,
                                 agile::workflow1::Edge, OECPolicy>
    Graph;

int main(int argc, char* argv[]) {
  galois::DistMemSys G; // init galois memory
  auto& net = galois::runtime::getSystemNetworkInterface();

  if (argc == 3)
    galois::setActiveThreads(atoi(argv[2]));

  if (net.ID == 0) {
    galois::gPrint("Testing building WMD graph from file.\n");
    galois::gPrint("Num Hosts: ", net.Num,
                   ", Active Threads Per Hosts: ", galois::getActiveThreads(),
                   "\n");
  }

  std::string dataFile = argv[1];
  std::vector<std::string> filenames;
  filenames.emplace_back(dataFile);
  std::vector<std::unique_ptr<galois::graphs::FileParser<
      agile::workflow1::Vertex, agile::workflow1::Edge>>>
      parsers;
  parsers.emplace_back(
      std::make_unique<galois::graphs::WMDParser<agile::workflow1::Vertex,
                                                 agile::workflow1::Edge>>(
          10, filenames));
  I_INIT("tmp/WMDGraph", net.ID, net.Num, 0);
  Graph* graph = new Graph(parsers, net.ID, net.Num, true, false,
                           galois::graphs::BALANCED_EDGES_OF_MASTERS);
  I_ROUND(0);
  I_CLEAR();
  assert(graph != nullptr);

  // generate a file with sorted token of all nodes and its outgoing edge dst
  // compare it with other implementation to verify the correctness
  std::vector<std::pair<uint64_t, std::vector<uint64_t>>> tokenAndEdges;
  tokenAndEdges.resize(graph->numMasters());

  galois::do_all(
      galois::iterate(graph->masterNodesRange()),
      [&](size_t lid) {
        auto token = graph->getData(lid).id;

        std::vector<uint64_t> edgeDst;
        auto end = graph->edge_end(lid);
        auto itr = graph->edge_begin(lid);
        for (; itr != end; itr++) {
          edgeDst.push_back(graph->getEdgeData(itr).dst);
        }
        std::sort(edgeDst.begin(), edgeDst.end());

        tokenAndEdges[lid] = std::make_pair(token, std::move(edgeDst));
      },
      galois::steal());

  // gather node info from other hosts
  if (net.ID != 0) { // send token and degree pairs to host 0
    galois::runtime::SendBuffer sendBuffer;
    galois::runtime::gSerialize(sendBuffer, tokenAndEdges);
    net.sendTagged(0, galois::runtime::evilPhase, sendBuffer);
  } else { // recv node range from other hosts
    for (size_t i = 0; i < net.Num - 1; i++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      std::vector<std::pair<uint64_t, std::vector<uint64_t>>>
          incomingtokenAndEdges;
      galois::runtime::gDeserialize(p->second, incomingtokenAndEdges);

      // combine data
      std::move(incomingtokenAndEdges.begin(), incomingtokenAndEdges.end(),
                std::back_inserter(tokenAndEdges));
    }
  }

  // sort the node info by token order
  // serilize it to file
  if (net.ID == 0) {
    std::sort(tokenAndEdges.begin(), tokenAndEdges.end(),
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a,
                 const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                return a.first < b.first;
              });

    std::ofstream output;
    output.open("wmd-graph-build-result.txt");

    for (auto itr : tokenAndEdges) {
      output << itr.first;
      for (auto edge : itr.second) {
        output << "," << edge;
      }
      output << "\n";
    }
    output.close();
  }
  I_DEINIT();
  return 0;
}
