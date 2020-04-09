#include "InpWriter.h"

#include "../model/Graph.h"
#include "../model/NodeData.h"
#include "../utils/ConnectivityManager.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using inpEdge = std::pair<size_t, size_t>;

void inpWriter(const std::string filename, Graph& graph) {

  auto nodeVector  = std::vector<InpNodeInfo>{};
  auto conecVector = std::vector<InpConecInfo>{};

  // Process the graph and get the vectors to write the inp file
  processGraph(graph, nodeVector, conecVector);

  // Write the file
  auto file = std::ofstream(filename);

  if (!file.is_open()) {
    std::cerr << "Cannot open output file " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // Write header
  file << nodeVector.size() << " " << nodeVector.size() + conecVector.size()
       << " 0 0 0" << std::endl;

  // Write nodes
  auto counter = 0;
  for (const auto node : nodeVector) {
    file << counter << " " << node.coods.getX() << " " << node.coods.getY()
         << " " << node.coods.getZ() << std::endl;

    ++counter;
  }

  // Write elements
  // First elements related to nodes (points)
  counter = 0;
  for (const auto node : nodeVector) {
    file << counter << " " << node.mat << " pt " << node.id << std::endl;

    ++counter;
  }

  // Then elements related to nodes (interior, edges, and triangles)
  for (const auto conec : conecVector) {
    file << counter << " " << conec.mat << " " << conec.type;

    for (const auto id : conec.conec) {
      file << " " << id;
    }

    file << std::endl;
    ++counter;
  }

  file.close();
}

void processGraph(Graph& graph, std::vector<InpNodeInfo>& nodeVector,
                  std::vector<InpConecInfo>& conecVector) {
  size_t nodeCounter = 0;

  auto connManager = ConnectivityManager{graph};
  auto nodeMap     = std::map<GNode, size_t>{};

  // First, process mesh nodes
  for (const auto graphNode : graph) {
    if (!graphNode->getData().isHyperEdge()) { // Only mesh nodes
      const auto coords = graphNode->getData().getCoords();
      const auto mat    = (graphNode->getData().isHanging()) ? 1u : 0u;
      const auto id     = nodeCounter;
      nodeVector.push_back(InpNodeInfo{coords, mat, id});
      nodeMap.insert({graphNode, id});
      ++nodeCounter;
    }
  }

  auto edgeSet = std::set<inpEdge>{};

  // Then, we process interiors
  for (const auto graphNode : graph) {
    if (!graphNode->getData().isHyperEdge()) {
      continue;
    }

    const auto coords = graphNode->getData().getCoords();
    const auto mat    = (graphNode->getData().isToRefine()) ? 3u : 2u;
    const auto id     = nodeCounter;
    nodeVector.push_back(InpNodeInfo{coords, mat, id});
    nodeMap.insert({graphNode, id});
    ++nodeCounter;

    // Get the three mesh node Ids

    const auto intNodes = connManager.getNeighbours(graphNode);
    const auto intNodesID =
        std::vector<size_t>{nodeMap.at(intNodes[0]), nodeMap.at(intNodes[1]),
                            nodeMap.at(intNodes[2])};

    // Now we generate the mesh triangle
    conecVector.push_back(InpConecInfo{intNodesID, 7, "tri"});

    // Now we generate the three edges that join the interior and the mesh
    // nodes
    conecVector.push_back(
        InpConecInfo{std::vector<size_t>{id, intNodesID.at(0)}, 4, "line"});

    conecVector.push_back(
        InpConecInfo{std::vector<size_t>{id, intNodesID.at(1)}, 4, "line"});

    conecVector.push_back(
        InpConecInfo{std::vector<size_t>{id, intNodesID.at(2)}, 4, "line"});

    // Finally, we generate the triangle edges
    for (auto i = 0u; i < intNodesID.size(); ++i) {

      // Create the inp edge
      const auto edge =
          inpEdge{std::min(intNodesID[i], intNodesID[(i + 1) % 3]),
                  std::max(intNodesID[i], intNodesID[(i + 1) % 3])};

      // Check if it has been created before
      if (edgeSet.insert(edge).second) {

        // Get the graph edge to see if it's on the boundary
        const auto graphEdge =
            connManager.getEdge(intNodes[i], intNodes[(i + 1) % 3]);

        const auto mat =
            connManager.getGraph().getEdgeData(graphEdge.get()).isBorder() ? 6u
                                                                           : 5u;

        conecVector.push_back(InpConecInfo{
            std::vector<size_t>{edge.first, edge.second}, mat, "line"});
      }
    }
  }
}
