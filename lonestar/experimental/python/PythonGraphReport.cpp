#include "PythonGraph.h"

#include <iostream>
#include <fstream>

unsigned rightmostSetBitPos(uint32_t n) {
  assert(n != 0);
  if (n & 1) return 0;

  // unset rightmost bit and xor with itself
  n = n ^ (n & (n - 1));

  unsigned pos = 0;
  while (n) {
    n >>= 1;
    pos++;
  }
  return pos-1;
}

void reportGraphSimulation(AttributedGraph& qG, AttributedGraph& dG, char* outputFile) {
  std::streambuf* buf;
  std::ofstream ofs;

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  Graph& qgraph = qG.graph;
  auto& qnodeNames = qG.nodeNames;
  Graph& graph = dG.graph;
  auto& nodeLabelNames = dG.nodeLabelNames;
  auto& edgeLabelNames = dG.edgeLabelNames;
  auto& nodeNames = dG.nodeNames;
  for(auto n: graph) {
    auto& src = graph.getData(n);
    auto& srcLabel = nodeLabelNames[rightmostSetBitPos(src.label)];
    auto& srcName = nodeNames[src.id];
    for(auto e: graph.edges(n)) {
      auto& dst = graph.getData(graph.getEdgeDst(e));
      auto& dstLabel = nodeLabelNames[rightmostSetBitPos(dst.label)];
      auto& dstName = nodeNames[dst.id];
      auto& ed = graph.getEdgeData(e);
      auto& edgeLabel = edgeLabelNames[rightmostSetBitPos(ed.label)];
      auto& edgeTimestamp = ed.timestamp;
      for(auto qn: qgraph) {
        uint64_t mask = (1 << qn);
        if (src.matched & mask) {
          for(auto qe: qgraph.edges(qn)) {
            auto& qeData = qgraph.getEdgeData(qe);
            if (qeData.label & ed.label) { // query could be any or multiple labels
              auto qDst = qgraph.getEdgeDst(qe);
              mask = (1 << qDst);
              if (dst.matched & mask) {
                auto& qSrcName = qnodeNames[qgraph.getData(qn).id];
                auto& qDstName = qnodeNames[qgraph.getData(qDst).id];
                os << srcLabel << " " << srcName << " ("  << qSrcName << ") "
                   << edgeLabel << " " << dstLabel << " "
                   << dstName << " ("  << qDstName << ") "
                   << " at " << edgeTimestamp << std::endl;
                break;
              }
            }
          }
        }
      }
    }
  }

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.close();
  }
}

void returnMatchedNodes(AttributedGraph& dataGraph, MatchedNode* matchedNodes) {
  Graph& graph = dataGraph.graph;
  //auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  size_t i = 0;
  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      matchedNodes[i].id = data.id;
      //matchedNodes[i].label = nodeLabelNames[data.label].c_str();
      matchedNodes[i].name = nodeNames[n].c_str();
      ++i;
    }
  }
}

void reportMatchedNodes(AttributedGraph &dataGraph, char* outputFile) {
  Graph& graph = dataGraph.graph;
  auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  std::streambuf* buf;
  std::ofstream ofs;

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      os << nodeLabelNames[data.label] << " " << nodeNames[n] << std::endl;
    }
  }

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.close();
  }
}

void returnMatchedNeighbors(AttributedGraph& dataGraph, uint32_t uuid, MatchedNode* matchedNeighbors) {
  Graph& graph = dataGraph.graph;
  //auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  size_t i = 0;
  // do not include the same node twice (multiple edges to the same node)
  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      matchedNeighbors[i].id = data.id;
      //matchedNeighbors[i].label = nodeLabelNames[data.label].c_str();
      matchedNeighbors[i].name = nodeNames[n].c_str();
      ++i;
    }
  }
}

void reportMatchedNeighbors(AttributedGraph &dataGraph, uint32_t uuid, char* outputFile) {
  Graph& graph = dataGraph.graph;
  auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  std::streambuf* buf;
  std::ofstream ofs;

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  // do not include the same node twice (multiple edges to the same node)
  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      os << nodeLabelNames[data.label] << " " << nodeNames[n] << std::endl;
    }
  }

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.close();
  }
}

void returnMatchedEdges(AttributedGraph& g, MatchedEdge* matchedEdges) {
  Graph& graph = g.graph;
  //auto& nodeLabelNames = g.nodeLabelNames;
  auto& edgeLabelNames = g.edgeLabelNames;
  auto& nodeNames = g.nodeNames;
  auto sourceLabelID = g.nodeLabelIDs["process"];

  size_t i = 0;
  for(auto src: graph) {
    auto& srcData = graph.getData(src);
    if (!srcData.matched) continue;
    //if ((srcData.label != sourceLabelID) || !srcData.matched) continue;
    //auto& srcLabel = nodeLabelNames[srcData.label];
    for(auto e: graph.edges(src)) {
      auto eData = graph.getEdgeData(e);
      if (eData.matched) {
        auto dst = graph.getEdgeDst(e);
        auto& dstData = graph.getData(dst);
        //if ((dstData.label == sourceLabelID) && (dst < src)) continue;
        //auto& dstLabel = nodeLabelNames[dstData.label];
        matchedEdges[i].timestamp = eData.timestamp;
        matchedEdges[i].label = edgeLabelNames[eData.label].c_str();
        if ((dstData.label != sourceLabelID) || ((srcData.label == sourceLabelID) && (src < dst))) {
          matchedEdges[i].caused_by.id = srcData.id;
          matchedEdges[i].caused_by.name = nodeNames[src].c_str();
          matchedEdges[i].acted_on.id = dstData.id;
          matchedEdges[i].acted_on.name = nodeNames[dst].c_str();
        } else {
          matchedEdges[i].caused_by.id = dstData.id;
          matchedEdges[i].caused_by.name = nodeNames[dst].c_str();
          matchedEdges[i].acted_on.id = srcData.id;
          matchedEdges[i].acted_on.name = nodeNames[src].c_str();
        }
        ++i;
      }
    }
  }
}

void reportMatchedEdges(AttributedGraph& g, char* outputFile) {
  Graph& graph = g.graph;
  //auto& nodeLabelNames = g.nodeLabelNames;
  auto& edgeLabelNames = g.edgeLabelNames;
  auto& nodeNames = g.nodeNames;
  auto sourceLabelID = g.nodeLabelIDs["process"];

  std::streambuf* buf;
  std::ofstream ofs;

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  for(auto src: graph) {
    auto& srcData = graph.getData(src);
    if (!srcData.matched) continue;
    //if ((srcData.label != sourceLabelID) || !srcData.matched) continue;
    //auto& srcLabel = nodeLabelNames[srcData.label];
    auto& srcName = nodeNames[src];
    for(auto e: graph.edges(src)) {
      auto eData = graph.getEdgeData(e);
      if (eData.matched) {
        auto dst = graph.getEdgeDst(e);
        auto& dstData = graph.getData(dst);
        //if ((dstData.label == sourceLabelID) && (dst < src)) continue;
        //auto& dstLabel = nodeLabelNames[dstData.label];
        auto& dstName = nodeNames[dst];
        auto& edgeLabel = edgeLabelNames[eData.label];
        auto& edgeTimestamp = eData.timestamp;
        if ((dstData.label != sourceLabelID) || ((srcData.label == sourceLabelID) && (src < dst))) {
          os << edgeTimestamp << ", " << srcName << ", "
                    << edgeLabel << ", " << dstName << std::endl;
        } else {
          os << edgeTimestamp << ", " << dstName << ", "
                    << edgeLabel << ", " << srcName << std::endl;
        }
      }
    }
  }

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.close();
  }
}

void returnMatchedNeighborEdges(AttributedGraph& g, uint32_t uuid, MatchedEdge* matchedEdges) {
  Graph& graph = g.graph;
  //auto& nodeLabelNames = g.nodeLabelNames;
  auto& edgeLabelNames = g.edgeLabelNames;
  auto& nodeNames = g.nodeNames;
  auto sourceLabelID = g.nodeLabelIDs["process"];
  auto src = g.nodeIndices[uuid];

  size_t i = 0;
  auto& srcData = graph.getData(src);
  //auto& srcLabel = nodeLabelNames[srcData.label];
  for(auto e: graph.edges(src)) {
    auto dst = graph.getEdgeDst(e);
    auto& dstData = graph.getData(dst);
    if (dstData.matched) {
      //auto& dstLabel = nodeLabelNames[dstData.label];
      auto& eData = graph.getEdgeData(e);
      matchedEdges[i].timestamp = eData.timestamp;
      matchedEdges[i].label = edgeLabelNames[eData.label].c_str();
      if ((dstData.label != sourceLabelID) || ((srcData.label == sourceLabelID) && (src < dst))) {
        matchedEdges[i].caused_by.id = srcData.id;
        matchedEdges[i].caused_by.name = nodeNames[src].c_str();
        matchedEdges[i].acted_on.id = dstData.id;
        matchedEdges[i].acted_on.name = nodeNames[dst].c_str();
      } else {
        matchedEdges[i].caused_by.id = dstData.id;
        matchedEdges[i].caused_by.name = nodeNames[dst].c_str();
        matchedEdges[i].acted_on.id = srcData.id;
        matchedEdges[i].acted_on.name = nodeNames[src].c_str();
      }
      ++i;
    }
  }
}

void reportMatchedNeighborEdges(AttributedGraph& g, uint32_t uuid, char* outputFile) {
  Graph& graph = g.graph;
  //auto& nodeLabelNames = g.nodeLabelNames;
  auto& edgeLabelNames = g.edgeLabelNames;
  auto& nodeNames = g.nodeNames;
  auto sourceLabelID = g.nodeLabelIDs["process"];
  auto src = g.nodeIndices[uuid];

  std::streambuf* buf;
  std::ofstream ofs;

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  auto& srcData = graph.getData(src);
  //auto& srcLabel = nodeLabelNames[srcData.label];
  auto& srcName = nodeNames[src];
  for(auto e: graph.edges(src)) {
    auto dst = graph.getEdgeDst(e);
    auto& dstData = graph.getData(dst);
    if (dstData.matched) {
      //auto& dstLabel = nodeLabelNames[dstData.label];
      auto& dstName = nodeNames[dst];
      auto& ed = graph.getEdgeData(e);
      auto& edgeLabel = edgeLabelNames[ed.label];
      auto& edgeTimestamp = ed.timestamp;
      if ((dstData.label != sourceLabelID) || ((srcData.label == sourceLabelID) && (src < dst))) {
        os << edgeTimestamp << ", " << srcName << ", "
                  << edgeLabel << ", " << dstName << std::endl;
      } else {
        os << edgeTimestamp << ", " << dstName << ", "
                  << edgeLabel << ", " << srcName << std::endl;
      }
    }
  }

  if ((outputFile != NULL) && (strcmp(outputFile, "") != 0)) {
    ofs.close();
  }
}
