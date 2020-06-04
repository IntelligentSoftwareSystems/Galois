#include "TriangleFormatWriter.h"

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

using triEdge = std::pair<size_t, size_t>;

void triangleFormatWriter(const std::string& filename, Graph& graph) {

  auto nodeVector  = std::vector<TriNodeInfo>{};
  auto segmVector  = std::vector<TriSegmInfo>{};
  auto conecVector = std::vector<TriConecInfo>{};

  // Process the graph and get the vectors to write the tri file
  trProcessGraph(graph, nodeVector, segmVector, conecVector);

  // Write the file
  auto nodeFile = std::ofstream(filename + ".node");

  if (!nodeFile.is_open()) {
    std::cerr << "Cannot open output file " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // Write header
  nodeFile << nodeVector.size() << " 2 1 0" << std::endl;

  // Write nodes
  auto counter = 0;
  for (const auto node : nodeVector) {
    nodeFile << counter++ << " " << node.coods.getX() << " "
             << node.coods.getY() << " " << node.coods.getZ() << std::endl;
  }
  nodeFile.close();

  // Write elements
  auto eleFile = std::ofstream(filename + ".ele");

  if (!eleFile.is_open()) {
    std::cerr << "Cannot open output file " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // Write header
  eleFile << conecVector.size() << " 3 0" << std::endl;

  // First elements related to nodes (points)
  counter = 0;

  // Then elements related to eles (edges and triangles)
  for (const auto& conec : conecVector) {
    eleFile << counter++;

    for (const auto id : conec.conec) {
      eleFile << " " << id;
    }

    eleFile << std::endl;
  }

  eleFile.close();

  // Write elements
  auto polyFile = std::ofstream(filename + ".poly");

  if (!polyFile.is_open()) {
    std::cerr << "Cannot open output file " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  polyFile << nodeVector.size() << " 2 1 0" << std::endl;

  //.poly file
  // Write nodes
  counter = 0;
  for (const auto node : nodeVector) {
    polyFile << counter++ << " " << node.coods.getX() << " "
             << node.coods.getY() << " " << node.coods.getZ() << std::endl;
  }

  counter = 0;
  polyFile << segmVector.size() << " 1" << std::endl;
  for (const auto& segm : segmVector) {
    polyFile << counter++ << " " << segm.points[0] << " " << segm.points[1]
             << " " << (segm.border ? 1 : 0) << std::endl;
  }

  polyFile << "0" << std::endl;

  polyFile.close();
}
void trProcessGraph(Graph& graph, std::vector<TriNodeInfo>& nodeVector,
                    std::vector<TriSegmInfo>& segmVector,
                    std::vector<TriConecInfo>& conecVector) {
  size_t nodeCounter = 0;

  auto connManager = ConnectivityManager{graph};
  auto nodeMap     = std::map<GNode, size_t>{};

  // First, process mesh nodes
  for (const auto graphNode : graph) {
    if (!graphNode->getData().isHyperEdge()) { // Only mesh nodes
      const auto coords = graphNode->getData().getCoords();
      const auto mat    = (graphNode->getData().isHanging()) ? 1u : 0u;
      const auto id     = nodeCounter;
      nodeVector.push_back(TriNodeInfo{coords, mat, id});
      nodeMap.insert({graphNode, id});
      ++nodeCounter;
    }
  }

  auto edgeSet = std::set<triEdge>{};

  // Then, we process interiors
  for (const auto graphNode : graph) {
    if (!graphNode->getData().isHyperEdge()) {
      continue;
    }

    const auto id = nodeCounter;
    nodeMap.insert({graphNode, id});
    ++nodeCounter;

    // Get the three mesh node Ids
    const auto intNodes = connManager.getNeighbours(graphNode);
    const auto intNodesID =
        std::vector<size_t>{nodeMap.at(intNodes[0]), nodeMap.at(intNodes[1]),
                            nodeMap.at(intNodes[2])};
    changeOrientationIfRequired(intNodesID, nodeVector);

    // Now we generate the mesh triangle
    conecVector.push_back(TriConecInfo{intNodesID, 7});

    // Finally, we generate the triangle edges
    for (auto i = 0u; i < intNodesID.size(); ++i) {

      // Create the tri edge
      const auto edge =
          triEdge{std::min(intNodesID[i], intNodesID[(i + 1) % 3]),
                  std::max(intNodesID[i], intNodesID[(i + 1) % 3])};

      // Check if it has been created before
      if (edgeSet.insert(edge).second) {

        // Get the graph edge to see if it's on the boundary
        const auto graphEdge =
            connManager.getEdge(intNodes[i], intNodes[(i + 1) % 3]);

        segmVector.push_back(TriSegmInfo{
            std::vector<size_t>{edge.first, edge.second},
            connManager.getGraph().getEdgeData(graphEdge.get()).isBorder()});
      }
    }
  }
}

void changeOrientationIfRequired(std::vector<unsigned long> element,
                                 std::vector<TriNodeInfo> nodeVector) {
  if (greater(((nodeVector[element[1]].coods.getX() -
                nodeVector[element[0]].coods.getX()) *
               (nodeVector[element[2]].coods.getY() -
                nodeVector[element[0]].coods.getY())) -
                  ((nodeVector[element[1]].coods.getY() -
                    nodeVector[element[0]].coods.getY()) *
                   (nodeVector[element[2]].coods.getX() -
                    nodeVector[element[0]].coods.getX())),
              0.)) {
    std::iter_swap(element.begin() + 1, element.begin() + 2);
  }
}
