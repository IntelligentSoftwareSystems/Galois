#ifndef GALOIS_MYGRAPHFORMATWRITER_H
#define GALOIS_MYGRAPHFORMATWRITER_H

#include <fstream>
#include <utility>

using std::set;
using std::string;
typedef std::tuple<string, string, bool> Edge;
typedef std::pair<int, NodeData> Node;

class MyGraphFormatWriter {
private:
  static void writeToMyGraphFormat(const set<Node>& vertices,
                                   const set<Node>& interiors,
                                   const set<Edge>& edges, const string& path) {
    std::ofstream file;
    file.open(path);
    printNodes(
        vertices, "N,n", [](NodeData n) { return n.isHanging(); }, file);
    printNodes(
        interiors, "H,h", [](NodeData n) { return n.isToRefine(); }, file);
    printEdges(edges, file);
    file.close();
  }

  static void printEdges(const set<Edge>& edges, std::ofstream& file) {
    int i = 0;
    for (auto edge : edges) {
      file << "E,e" << i++ << "," << std::get<0>(edge) << ","
           << std::get<1>(edge) << "," << (std::get<2>(edge) ? "true" : "false")
           << std::endl;
    }
  }

  static void printNodes(const set<Node>& nodes, const string preambule,
                         bool (*attributeChecker)(NodeData),
                         std::ofstream& file) {
    for (auto node : nodes) {
      file << preambule << node.first << "," << node.second.getCoords().getX()
           << "," << node.second.getCoords().getY() << ","
           << node.second.getCoords().getZ() << ","
           << (attributeChecker(node.second) ? "true" : "false") << std::endl;
    }
  }

  static void addEdge(set<Edge>& edges, const string& firstNodeId,
                      const string& secondNodeId, bool border) {
    if (!findEdge(firstNodeId, secondNodeId, edges).is_initialized()) {
      edges.emplace(std::make_tuple(firstNodeId, secondNodeId, border));
    }
  }

  static optional<Edge> findEdge(const string& first, const string& second,
                                 const set<Edge>& edges) {
    for (auto edge : edges) {
      if ((std::get<0>(edge) == first && std::get<1>(edge) == second) ||
          (std::get<0>(edge) == second && std::get<1>(edge) == first)) {
        return galois::optional<Edge>(edge);
      }
    }
    return galois::optional<Edge>();
  }

  static string getNodeId(set<Node>& nodes, int& nodesIter, NodeData& data) {
    return getNodeId(nodes, nodesIter, data, optional<set<Node>>());
  }

  static string getNodeId(set<Node>& nodes, int& nodesIter, NodeData& data,
                          optional<set<Node>> additionalNodesSet) {
    optional<string> maybeId = findNode(data, nodes);
    if (maybeId) {
      return maybeId.get();
    }
    if (additionalNodesSet) {
      optional<string> maybeId2 = findNode(data, additionalNodesSet.get());
      if (maybeId2) {
        return maybeId2.get();
      }
    }
    nodes.emplace(Node(nodesIter, data));
    return (data.isHyperEdge() ? "h" : "n") + std::to_string(nodesIter++);
  }

  static optional<string> findNode(const NodeData& node,
                                   const set<Node>& nodesSet) {
    for (auto pair : nodesSet) {
      if (pair.second == node) {
        return optional<string>((node.isHyperEdge() ? "h" : "n") +
                                std::__cxx11::to_string(pair.first));
      }
    }
    return optional<string>();
  }

public:
  static void writeToFile(Graph& graph, const string& path) {
    set<Node> vertices;
    set<Node> interiors;
    set<Edge> edges;
    int nodesIter     = 0;
    int interiorsIter = 0;
    for (auto node : graph) {
      NodeData& data = graph.getData(node);
      if (!data.isHyperEdge()) {
        string firstNodeId = getNodeId(vertices, nodesIter, data);
        for (const EdgeIterator& e : graph.edges(node)) {
          GNode dstNode        = graph.getEdgeDst(e);
          NodeData dstNodeData = graph.getData(dstNode);
          if (!dstNodeData.isHyperEdge()) {
            string secondNodeId = getNodeId(vertices, nodesIter, dstNodeData);
            addEdge(
                edges, firstNodeId, secondNodeId,
                graph.getEdgeData(graph.findEdge(node, dstNode)).isBorder());
          }
        }
      } else {
        string firstInteriorId = getNodeId(interiors, interiorsIter, data,
                                           optional<set<Node>>(vertices));
        for (const EdgeIterator& e : graph.edges(node)) {
          GNode dstNode        = graph.getEdgeDst(e);
          NodeData dstNodeData = graph.getData(dstNode);
          string secondNodeId = getNodeId(interiors, interiorsIter, dstNodeData,
                                          optional<set<Node>>(vertices));
          addEdge(edges, firstInteriorId, secondNodeId,
                  graph.getEdgeData(graph.findEdge(node, dstNode)).isBorder());
        }
      }
    }
    writeToMyGraphFormat(vertices, interiors, edges, path);
  }
};

#endif // GALOIS_MYGRAPHFORMATWRITER_H
