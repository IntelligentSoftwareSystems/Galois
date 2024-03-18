#ifndef LIBGALOIS_INCLUDE_SHAED_GRAPH_READER_H_
#define LIBGALOIS_INCLUDE_SHAED_GRAPH_READER_H_

#include <fstream>
#include <string>

#include "galois/graphs/BufferedGraph.h"

#include "shad/DataTypes.h"
#include "shad/Graph.h"
#include "shad/GraphTypes.h"

namespace shad {

struct ShadNodeTy {
  int type;
  uint64_t key;
};
using ShadEdgeTy = uint64_t;

/**
 * TODO(hc): This is a shared-memory version.
 * Later, a distributed-memory version in libgluon will reuse this code.
 */
class ShadGraphConverter {

public:
  ShadGraphConverter() : nodeDataBuffer(nullptr) {}

  ~ShadGraphConverter() {
    // BufferedGraph holds these arrays.
    outIndexBuffer = nullptr;
    nodeDataBuffer = nullptr;
    edgeDestBuffer = nullptr;
    edgeDataBuffer = nullptr;
  }

  /**
   * @brief Flush a graph topology to a file for debugging.
   */
  void flushGraphTopology() {
    std::ofstream fp("shad_graph.out");
    for (size_t i = 0; i < this->verticeIdKeyMapping.size(); ++i) {
      uint64_t key = this->verticeIdKeyMapping[i];
      Vertex v     = this->vertices[key];
      fp << "node " << i << ", type: " << to_underlying(v.type)
         << ", key: " << key << "\n";
      auto edgeRange = this->edges.equal_range(key);
      for (auto ei = edgeRange.first; ei != edgeRange.second; ++ei) {
        Edge& edge = ei->second;
        Vertex dst = this->vertices[edge.dst];
        fp << "\t edge dst " << dst.id << ", type: " << to_underlying(edge.type)
           << ", key: " << dst.shadKey << "\n";
      }
    }
    fp.close();
  }

  /**
   * @brief Read a input graph file and inspect the number of nodes and edges.
   * @detail In order to construct a dense LC_CSR_Graph, we need to know how
   * many edges and nodes exist. This method reads one line by one line, and
   * counts those information.
   * Note that this method assumes that the types of {"Person", "ForumEvent",
   * "Forum", "Publication", "Topic"} are nodes, and the types of
   * {"SALE", "Author", "Includes", "HasTopic", "HasOrg"} are edges.
   *
   * @param filename file name to read
   * @param numNodes number of nodes that this method reads
   * @param numEdges number of edges that this method reads
   */
  void InspectGraph(const std::string& filename, size_t* numNodes,
                    size_t* numEdges) {
    // TODO(hc): Get the number of nodes and edges from file
    // For example, it reads {SALE, Author, Includes, HasTopic, HasOrg} as
    // edges. So we just count how many they exist in the file.

    std::string line;
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Cannot open file " << filename << "\n";
      exit(-1);
    }
    while (!file.eof()) {
      getline(file, line);
      // Skip comments.
      if (line[0] == '#')
        continue;
      // Delimiter and # tokens set for WMD data file.
      std::vector<std::string> tokens = splitTokens(line, ',', 10);

      if (this->isTokenNodeType(tokens[0])) {
        ++(*numNodes);
      } else if (this->isTokenEdgeType(tokens[0])) {
        *numEdges += 2;
      }
    }

    std::cout << "Number of nodes:" << *numNodes
              << ", number of edges:" << *numEdges << "\n";
  }

  /**
   * @brief Construct a buffered graph from existing arrays constructed
   * by constructNodeArrays() and constructEdgeArrays().
   *
   * @param numGlobalNodes The number of global nodes
   * @param numGlobalEdges The number of global edges
   * @param nodeBegin Global node ID of the first local node
   * @param nodeEnd (Global node ID of the last local node) + 1
   * @param edgeBegin Global edge ID of the first local edge
   * @param edgeEnd (Global edge ID of the last local edge) + 1
   * @param bufferedGraph Buffered graph for CuSP
   */
  void constructBufferedGraph(
      uint64_t numGlobalNodes, uint64_t numGlobalEdges, uint32_t nodeBegin,
      uint32_t nodeEnd, uint64_t edgeBegin, uint64_t edgeEnd,
      [[maybe_unused]] galois::graphs::BufferedGraph<ShadEdgeTy>*
          bufferedGraph) {
    // TODO(hc): Each of these functions first construct graphs in the SHAD
    // format as this file is written in not binary, but string, and also
    // nodes or edges are not sorted. So, until we preprocess the input graph
    // file, we should first read it in memory, and reconstruct this to Galois
    // compatible

    uint32_t numLocalNodes = nodeEnd - nodeBegin;
    uint64_t numLocalEdges = edgeEnd - edgeBegin;

    bufferedGraph->constructFrom(outIndexBuffer, edgeDestBuffer, edgeDataBuffer,
                                 numGlobalNodes, numGlobalEdges, numLocalNodes,
                                 numLocalEdges, nodeBegin, edgeBegin);
#if 0
    TODO(hc): This verification should be fixed since it tests
              a shared-memory execution that one host loads the whole
              graph. It should not work on distributed-memory machine
              since a CSR graph should be partitioned but tepmorary
              maps reading and holding SHAD graphs are for global graph.
#ifndef NDEBUG
    std::cout << "CSR verification starts.." << std::endl << std::flush;
    this->VerifyCSRConstruction(outIndexBuffer, nodeDataBuffer,
        edgeDestBuffer, edgeDataBuffer);
    std::cout << "CSR verification starts.. [done]" << std::endl << std::flush;
#endif
#endif
    // TODO(hc): Construct `buffer_graph`.
  }

  /**
   * @brief Read SHAD graph file and construct in-memory buffer SHAD graph.
   *
   * @param filename SHAD graph file name
   */
  // TODO(hc): We can assign a disjointed range of file for each host.
  // For now, let all hosts read the whole file.
  void readSHADFile(const std::string& filename, uint64_t* numGlobalNodes,
                    uint64_t* numGlobalEdges) {
    std::ifstream graphFile(filename.c_str());
    uint64_t vertexId{0};
    std::string line;
    uint64_t numNodes{0}, numEdges{0};
    // TODO(hc): We can parallelize it by assigning disjointed
    // ranges with some inspection.
    // But this would be the future work as
    while (!graphFile.eof()) {
      getline(graphFile, line);
      // Skip comments.
      if (line[0] == '#')
        continue;
      // Delimiter and # tokens set for WMD data file.
      std::vector<std::string> tokens = splitTokens(line, ',', 10);

      if (tokens[0] == "Person") {
        insertSHADVertex(ENCODE<uint64_t, std::string, UINT>(tokens[1]),
                         TYPES::PERSON, vertexId);
        ++numNodes;
      } else if (tokens[0] == "ForumEvent") {
        insertSHADVertex(ENCODE<uint64_t, std::string, UINT>(tokens[4]),
                         TYPES::FORUMEVENT, vertexId);
        ++numNodes;
      } else if (tokens[0] == "Forum") {
        insertSHADVertex(ENCODE<uint64_t, std::string, UINT>(tokens[3]),
                         TYPES::FORUM, vertexId);
        ++numNodes;
      } else if (tokens[0] == "Publication") {
        insertSHADVertex(ENCODE<uint64_t, std::string, UINT>(tokens[5]),
                         TYPES::PUBLICATION, vertexId);
        ++numNodes;
      } else if (tokens[0] == "Topic") {
        insertSHADVertex(ENCODE<uint64_t, std::string, UINT>(tokens[6]),
                         TYPES::TOPIC, vertexId);
        ++numNodes;
      } else if (tokens[0] == "Sale") {
        Edge sale(tokens);
        insertSHADEdge(sale.src, sale);

        Edge purchase = sale;
        purchase.type = TYPES::PURCHASE;
        std::swap(purchase.src, purchase.dst);
        insertSHADEdge(purchase.src, purchase);
        numEdges += 2;
      } else if (tokens[0] == "Author") {
        Edge authors(tokens);
        insertSHADEdge(authors.src, authors);

        Edge writtenBY = authors;
        writtenBY.type = TYPES::WRITTENBY;
        std::swap(writtenBY.src, writtenBY.dst);
        std::swap(writtenBY.src_type, writtenBY.dst_type);
        insertSHADEdge(writtenBY.src, writtenBY);
        numEdges += 2;
      } else if (tokens[0] == "Includes") {
        Edge includes(tokens);
        insertSHADEdge(includes.src, includes);

        Edge includedIN = includes;
        includedIN.type = TYPES::INCLUDEDIN;
        std::swap(includedIN.src, includedIN.dst);
        std::swap(includedIN.src_type, includedIN.dst_type);
        insertSHADEdge(includedIN.src, includedIN);
        numEdges += 2;
      } else if (tokens[0] == "HasTopic") {
        Edge hasTopic(tokens);
        insertSHADEdge(hasTopic.src, hasTopic);

        Edge topicIN = hasTopic;
        topicIN.type = TYPES::TOPICIN;
        std::swap(topicIN.src, topicIN.dst);
        std::swap(topicIN.src_type, topicIN.dst_type);
        insertSHADEdge(topicIN.src, topicIN);
        numEdges += 2;
      } else if (tokens[0] == "HasOrg") {
        Edge hasOrg(tokens);
        insertSHADEdge(hasOrg.src, hasOrg);

        Edge orgIN = hasOrg;
        orgIN.type = TYPES::ORGIN;
        std::swap(orgIN.src, orgIN.dst);
        std::swap(orgIN.src_type, orgIN.dst_type);
        insertSHADEdge(orgIN.src, orgIN);
        numEdges += 2;
      }
    }

    // After the above loop, vertices and edges are complete.
    this->CountNumEdgesForEachVertex(numNodes, numEdges);
    *numGlobalNodes = numNodes;
    *numGlobalEdges = numEdges;

#ifndef NDEBUG
    this->VerifySHADGraphRead(filename);
#endif
  }

  /**
   * @brief Return node data array.
   * Note that this can be either of global graph or local graph.
   */
  ShadNodeTy* getNodeDataBuffer() { return nodeDataBuffer; }

  /**
   * @brief Return node outgoing edge index array
   * Note that this can be either of global graph or local graph.
   */
  uint64_t* getOutIndexBuffer() { return outIndexBuffer; }

  /**
   * @brief Construct vertex outgoing edge range buffer and
   * vertex data buffer.
   *
   * @detail Extract local vertices' outgoing edge ranges and
   * data from a temprory buffer of vertex map that is read and constructed
   * from a SHAD CSV graph file. Note that these arrays are for local graph
   * partition and their indices should be corresponding to local node ids.
   *
   * @param nodeBegin Global node ID of the first local node
   * @param nodeEnd (Global node ID of the last local node + 1)
   * @param numLocalNodes The number of local nodes
   *
   */
  void constructNodeArrays(uint32_t nodeBegin, uint32_t nodeEnd,
                           uint32_t numLocalNodes) {
    // 1) Construct an edge index array (size == number of nodes).
    this->outIndexBuffer = new uint64_t[numLocalNodes];
    this->nodeDataBuffer = new ShadNodeTy[numLocalNodes];

    // TODO(hc): for now, only consider a single host, but need to add offset
    // later.
    galois::do_all(galois::iterate(this->vertices), [&](auto element) {
      Vertex& vertex    = element.second;
      uint64_t vertexId = vertex.id;
      if (vertexId >= nodeBegin && vertexId < nodeEnd) {
        this->outIndexBuffer[vertexId - nodeBegin] = vertex.getNumEdges();
        // Fill vertex data too; This assumes that a SHAD graph
        // has a type, which is considered as a vertex data.
        this->nodeDataBuffer[vertexId - nodeBegin].type =
            this->to_underlying(vertex.type);
        this->nodeDataBuffer[vertexId - nodeBegin].key = vertex.shadKey;
        // std::cout << vertexId - nodeBegin << " is set to "
        //<< this->nodeDataBuffer[vertexId - nodeBegin].type << " and " <<
        // this->nodeDataBuffer[vertexId - nodeBegin].key << "\n";
      }
    });
    // 2) Perform parallel prefix sum to finalize outgoing edge index
    // array construction.
    galois::ParallelSTL::partial_sum(
        outIndexBuffer, &(outIndexBuffer[numLocalNodes]), outIndexBuffer);
  }

  /**
   * @brief Construct edge destination and data arrays.
   *
   * @detail Extract local edge destination and data from a
   * temprory buffer of edge map that is read and constructed
   * from a SHAD CSV graph file. Note that these arrays are for local graph
   * partition and their indices should be corresponding to local node ids.
   *
   * @tparam T Edge data type; if this is not void, edge data array is
   * constructed
   *
   * @param nodeBegin Global node ID of the first local node
   * @param edgeBegin Global edge ID of the first local edge
   * @param numLocalNodes The number of local nodes
   * @param numLocalEdges The number of local edges
   *
   */
  template <typename T                                           = ShadEdgeTy,
            typename std::enable_if_t<!std::is_same_v<T, void>>* = nullptr>
  void constructEdgeArrays(uint32_t nodeBegin, uint64_t edgeBegin,
                           uint32_t numLocalNodes, uint64_t numLocalEdges) {
    this->edgeDestBuffer = new uint32_t[numLocalEdges];
    this->edgeDataBuffer = new ShadEdgeTy[numLocalEdges];
    std::vector<uint32_t> edgeIndexPointers(numLocalNodes, 0);
    galois::on_each([&](uint32_t tid, uint32_t numThreads) {
      // 1) Find disjointed node range for each thread.
      auto thread_work_range =
          galois::block_range(uint32_t{0}, numLocalNodes, tid, numThreads);
      // 2) Each thread iterates the whole edges.
      for (auto edgeElem : this->edges) {
        uint64_t srcVertex   = edgeElem.first;
        Vertex& vertex       = this->vertices[srcVertex];
        uint64_t srcVertexId = vertex.id;
        // 3) Each thread fills edge destination for the assigned nodes.
        if (srcVertexId >= thread_work_range.first + nodeBegin &&
            srcVertexId < thread_work_range.second + nodeBegin) {
          uint64_t edgeIdx = edgeIndexPointers[srcVertexId - nodeBegin]++;
          // OutIndexBuffer now contains global edge range.
          // So we need to subtract edge offset to get the local edge id.
          uint64_t nodeBaseOffset =
              ((srcVertexId - nodeBegin) == 0)
                  ? 0
                  : outIndexBuffer[srcVertexId - nodeBegin - 1] - edgeBegin;
          edgeDestBuffer[edgeIdx + nodeBaseOffset] =
              this->vertices[edgeElem.second.dst].id;
          edgeDataBuffer[edgeIdx + nodeBaseOffset] =
              to_underlying(edgeElem.second.type);
        }
      }
    });
    // Or inspector/executor model
    // But that might be more expensive.
  }

  /**
   * @brief Construct edge destination array
   *
   * @detail Extract local edge destination from a
   * temprory buffer of edge map that is read and constructed
   * from a SHAD CSV graph file. Note that this array is for local graph
   * partition and their indices should be corresponding to local node ids.
   *
   * @tparam T Edge data type; This function is enabled when
   * edge data type is void
   *
   * @param nodeBegin Global node ID of the first local node
   * @param edgeBegin Global edge ID of the first local edge
   * @param numLocalNodes The number of local nodes
   * @param numLocalEdges The number of local edges
   *
   */
  template <typename T                                          = ShadEdgeTy,
            typename std::enable_if_t<std::is_same_v<T, void>>* = nullptr>
  void constructEdgeArrays(uint32_t nodeBegin, uint64_t edgeBegin,
                           uint32_t numLocalNodes, uint64_t numLocalEdges) {
    edgeDestBuffer = new uint32_t[numLocalEdges];
    std::vector<uint32_t> edgeIndexPointers(numLocalNodes, 0);
    galois::on_each([&](uint32_t tid, uint32_t numThreads) {
      // 1) Find disjointed node range for each thread.
      auto thread_work_range =
          galois::block_range(uint32_t{0}, numLocalNodes, tid, numThreads);
      // 2) Each thread iterates the whole edges.
      for (auto edgeElem : this->edges) {
        uint64_t srcVertex   = edgeElem.first;
        Vertex& vertex       = this->vertices[srcVertex];
        uint64_t srcVertexId = vertex.id;
        // 3) Each thread fills edge destination for the assigned nodes.
        if (srcVertexId >= thread_work_range.first + nodeBegin &&
            srcVertexId < thread_work_range.second + nodeBegin) {
          uint64_t edgeIdx = edgeIndexPointers[srcVertexId - nodeBegin]++;
          uint64_t nodeBaseOffset =
              ((srcVertexId - nodeBegin) == 0)
                  ? 0
                  : outIndexBuffer[srcVertexId - 1] - edgeBegin;
          edgeDestBuffer[edgeIdx + nodeBaseOffset] =
              this->vertices[edgeElem.second.dst].id;
        }
      }
    });
    // Or inspector/executor model
    // But that might be more expensive.
  }

  /**
   * @brief Extract outgoing edge index ranges for local vertices
   * from the global outgoing edge index range array.
   *
   * @param nodeBegin Node global id of the first local node
   * @param nodeEnd (Node global id for the last local node + 1)
   */
  void extractLocalOutIndexArray(uint32_t nodeBegin, uint32_t nodeEnd) {

    uint64_t* newOutIndexBuffer = new uint64_t[nodeEnd - nodeBegin];
    galois::do_all(galois::iterate(nodeBegin, nodeEnd), [&](uint32_t n) {
      newOutIndexBuffer[n - nodeBegin] = this->outIndexBuffer[n];
    });
    delete[] this->outIndexBuffer;
    this->outIndexBuffer = newOutIndexBuffer;
  }

  /**
   * @brief Check if a type of a node having the passed id is
   * equal to the one in a temporary vertex map constructed from
   * SHAD graph file.
   *
   * @param id Node global id to check
   * @param type Node type
   *
   * @return True if passed information matches to the one in
   * a temporary vertex map
   */
  bool checkNode(uint64_t id, int type) {
    uint64_t key   = this->verticeIdKeyMapping[id];
    Vertex& vertex = this->vertices[key];
    return (this->to_underlying(vertex.type) == type);
  }

  /**
   * @brief Check if a type of a edge having the passed id is
   * equal to the one in a temporary edge map constructed from
   * SHAD graph file.
   *
   * @param snid Global node ID of the source node of an edge
   * @param dnid Global node ID of the destination node of an edge
   * @param type Edge type
   * @param type Edge type
   *
   * @return True if passed information matches to the one in
   * a temporary edge map
   */
  bool checkEdge(uint64_t snid, uint64_t dnid, uint64_t /*eid*/, int type) {
    uint64_t skey  = this->verticeIdKeyMapping[snid];
    auto edgeRange = this->edges.equal_range(skey);
    uint64_t eidx{0};
    Edge edge;
    bool found{false};
    for (auto ei = edgeRange.first; ei != edgeRange.second; ++ei, ++eidx) {
      edge = ei->second;
      // Multiple edges having the same source and destination could
      // exist. So we repeat until find the one that has the same type to
      // the passed one.
      if (this->vertices[edge.dst].id == dnid &&
          this->to_underlying(edge.type) == type) {
        found = true;
        break;
      }
    }
    return found;
  }

private:
  /**
   * @brief Return true if a token is a node type.
   *
   * @param token Token parsed from a graph file to check
   */
  bool isTokenNodeType(std::string token) {
    if (token == "Person" || token == "ForumEvent" || token == "Forum" ||
        token == "Publication" || token == "Topic") {
      return true;
    } else {
      return false;
    }
  }

  /**
   * @brief Return true if a token is an edge type.
   *
   * @param token Token parsed from a graph file to check
   */
  bool isTokenEdgeType(std::string token) {
    if (token == "Sale" || token == "Author" || token == "Includes" ||
        token == "HasTopic" || token == "HasOrg") {
      return true;
    } else {
      return false;
    }
  }

  std::vector<std::string> splitTokens(std::string& line, char delim,
                                       uint64_t size = 0) {
    uint64_t ndx = 0, start = 0, end = 0;
    std::vector<std::string> tokens(size);

    for (; end < line.length(); end++) {
      if ((line[end] == delim) || (line[end] == '\n')) {
        tokens[ndx] = line.substr(start, end - start);
        start       = end + 1;
        ndx++;
      }
    }

    // Flush the last token.
    tokens[size - 1] = line.substr(start, end - start);
    return tokens;
  }

  void CountNumEdgesForEachVertex(uint64_t numNodes, uint64_t numEdges) {
    // galois::on_each([this, numNodes, numEdges](
    galois::on_each([&](uint32_t tid, uint32_t numThreads) {
      // Each thread is assigned disjointed range of nodes.
      // Each thread iterates edges and accumulates edges for only
      // the nodes assigned to that.
      auto thread_work_range =
          galois::block_range(uint64_t{0}, numNodes, tid, numThreads);
      for (auto edgeElem : this->edges) {
        uint64_t srcVertex = edgeElem.first;
        Vertex& vertex     = this->vertices[srcVertex];
        if (vertex.id >= thread_work_range.first &&
            vertex.id < thread_work_range.second) {
          vertex.incrNumEdges();
        }
      }
    });

#ifndef NDEBUG
    this->VerifyNumEdgesPerVertex(numEdges);
#else
    (void)numEdges;
#endif
  }

  /**
   * @brief Insert SHAD vertex to a vertex map.
   *
   * @param key SHAD token key
   * @param type SHAD vertex type
   * @param id Vertex id; Local vertex id until it is synchronized
   */
  void insertSHADVertex(const uint64_t& key, const TYPES& type, uint64_t& id) {
    auto found = this->vertices.find(key);
    if (found == this->vertices.end()) {
      this->vertices[key]           = Vertex(id, type, key);
      this->verticeIdKeyMapping[id] = key;
      id++;
    } else {
      std::cerr << "[error] There is no reason to have duplicated vertices\n";
    }
  }

  /**
   * @brief Insert SHAD edge to a edge map.
   * @detail Edges
   *
   * @param vertexKey Source vertex's SHAD token key
   * @param edge Adjacent edge of the vertex
   */
  void insertSHADEdge(const uint64_t& vertexKey, const Edge& edge) {
    this->edges.insert({vertexKey, edge});
  }

  /*
  uint64_t edge_begin(uint32_t n) {
    return this->verticeIdKeyMapping[n]
  */

#ifndef NDEBUG
  /**
   * @brief Verify in-meomry SHAD graph.
   *
   * @param filename SHAD graph file name
   */
  // TODO(hc): This function can be parallelized but
  // let me stick with sequential execution until the whole
  // implementation works correctly.
  void VerifySHADGraphRead(const std::string& filename) {
    size_t numNodes{0}, numEdges{0};
    this->InspectGraph(filename, &numNodes, &numEdges);
    // 1) Check the number of vertices and edges.
    assert(this->vertices.size() == numNodes);
    // Note that edges are doubled to symmetrize a graph.
    assert(this->edges.size() == numEdges);
    for ([[maybe_unused]] auto& element : this->edges) {
      // 2) Check if a source node key of the edges map is equal to a source
      // of an edge.
      assert(element.first == element.second.src);
      // 3) Check if vertex information in the edges map is equal to the one
      // in the vertex map.
      assert(element.second.src_type ==
             this->vertices[element.second.src].type);
      assert(element.second.dst_type ==
             this->vertices[element.second.dst].type);
    }
  }

  void VerifyNumEdgesPerVertex([[maybe_unused]] uint64_t numEdges) {
    // 4) Check if the total number of edges of each vertex is equal to
    // the number of total edges counted during inspection.
    uint64_t numAccumulatedEdges{0};
    for (auto& element : this->vertices) {
      numAccumulatedEdges += element.second.getNumEdges();
    }
    assert(numAccumulatedEdges == numEdges);
  }

  void VerifyCSRConstruction([[maybe_unused]] uint64_t* outIndexBuffer,
                             [[maybe_unused]] ShadNodeTy* nodeDataBuffer,
                             [[maybe_unused]] uint32_t* edgeDestBuffer,
                             [[maybe_unused]] void* edgeDataBuffer) {}

  template <typename T = ShadEdgeTy,
            typename std::enable_if_t<std::is_same_v<T, uint64_t>>* = nullptr>
  void VerifyCSRConstruction(uint64_t* outIndexBuffer,
                             [[maybe_unused]] ShadNodeTy* nodeDataBuffer,
                             uint32_t* edgeDestBuffer,
                             ShadEdgeTy* edgeDataBuffer) {
    // 1) Iterate edge index array.
    // 2) Compare each verteices' edge range with SHAD vertex
    for (size_t i = 0; i < this->vertices.size(); ++i) {
      Vertex& srcV        = this->vertices[this->verticeIdKeyMapping[i]];
      uint64_t srcShadKey = srcV.shadKey;
      assert(this->verticeIdKeyMapping[i] == srcV.shadKey);
      uint64_t edgeBegin = (i == 0) ? 0 : outIndexBuffer[i - 1];
      uint64_t edgeEnd   = outIndexBuffer[i];
      assert(srcV.numEdges == edgeEnd - edgeBegin);
      assert(this->to_underlying(srcV.type) == int(nodeDataBuffer[i].type));
      assert(srcV.id == i);
      galois::do_all(
          galois::iterate(edgeBegin, edgeEnd),
          [&](size_t j) {
            uint32_t dstV                      = edgeDestBuffer[j];
            [[maybe_unused]] uint64_t edgeData = edgeDataBuffer[j];

            [[maybe_unused]] bool found{false};
            auto edgeRange = this->edges.equal_range(srcShadKey);
            size_t cnt{0};
            for (auto ei = edgeRange.first; ei != edgeRange.second; ++ei) {
              Edge& edge = ei->second;
              if (this->vertices[edge.dst].id == dstV) {
                // Multiple edges between vertices are possible.
                if (this->to_underlying(edge.type) == int(edgeData)) {
                  assert(this->vertices[edge.src].id == i);
                  assert(this->vertices[edge.src].id == srcV.id);
                  found = true;
                }
              }
              cnt++;
            }
            assert((edgeEnd - edgeBegin) == cnt);
            /*
            for (auto i = this->edges.begin(); i != this->edges.end(); ++i) {
              std::cout << srcId << " vs " << i->first << "\n";
            }
            */
            assert(found);
          },
          galois::steal());
    }
  }
#endif

  /**
   * @brief Cast a type to an underlying type; in case of scoped enum,
   * this should be an integral type.
   *
   * @param e
   */
  template <typename E>
  constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
    return static_cast<typename std::underlying_type<E>::type>(e);
  }

  // This holds the whole global vertices and their
  // information such as its type. A key is globla node ID, and its value
  // is the information.
  std::unordered_map<uint64_t, Vertex> vertices;
  // This holds the whole global edges and their information
  // such as its type. The key is global source node ID, and its
  // value is an edge iterator pointing to adjacent edges to the source.
  std::unordered_multimap<uint64_t, Edge> edges;
  // Key is global node id and value is corresponding key of that node
  std::unordered_map<uint64_t, uint64_t> verticeIdKeyMapping;
  // TODO(hc): Always assume uint64_t node data type
  ShadNodeTy* nodeDataBuffer;
  uint64_t* outIndexBuffer;
  uint32_t* edgeDestBuffer;
  ShadEdgeTy* edgeDataBuffer;
};

}; // namespace shad

#endif
