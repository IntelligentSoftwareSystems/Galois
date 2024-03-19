/**
 * @file WMDGraph.h
 *
 * Contains the implementation of EdgeListBufferedGraph and EdgeListOfflineGraph which is
 * a galois graph constructed from WMD dataset
 */

#ifndef WMD_BUFFERED_GRAPH_H
#define WMD_BUFFERED_GRAPH_H

#include <fstream>
#include <unordered_map>
#include <atomic>
#include <cstring>
#include <cmath>
#include <iterator>
#include <sys/stat.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/serialization/vector.hpp>

#include "galois/runtime/Network.h"
#include "galois/config.h"
#include "galois/gIO.h"
#include "galois/Reduction.h"

#include "schema.h"

namespace galois {
namespace graphs {

void inline increment_evilPhase() {
  ++galois::runtime::evilPhase;
  if (galois::runtime::evilPhase >=
      static_cast<uint32_t>(
          std::numeric_limits<int16_t>::max())) { // limit defined by MPI or
                                                  // LCI
    galois::runtime::evilPhase = 1;
  }
}

/**
 * Load a graph from file in Edgelist format into memory.
 *
 * Inherit from OffilineGraph only to make it compatible with Partitioner
 * Internal implementation are completed different.
 */
template <typename NodeDataType, typename EdgeDataType>
class EdgeListOfflineGraph : public OfflineGraph {
protected:
  // TODO: consider typedef uint64_t NodeIDType ?
  typedef boost::counting_iterator<uint64_t> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;

  // global feilds (same on each hosts)
  std::vector<uint64_t> nodeOffset; // each hosts' local ID offset wrt global ID

  // per thread data struct (will be combined into a single data struct)
  std::vector<std::unordered_map<uint64_t, size_t>>
      perThreadTokenToLocalEdgesIdx;
  std::vector<std::vector<std::vector<EdgeDataType>>> perThreadLocalEdges;

  uint32_t hostID;
  uint32_t numHosts;

  /**
   * @brief this releases memory by swapping p_container with an empty container
   * and so, by using out-of-scope
   */
  template <typename T>
  static inline void freeContainer(T& p_container) {
    T empty;
    std::swap(p_container, empty);
  }

  void insertlocalEdgesPerThread(unsigned tid, uint64_t token,
                                        EdgeDataType& edge) {
    if (auto search = perThreadTokenToLocalEdgesIdx[tid].find(token);
        search !=
        perThreadTokenToLocalEdgesIdx[tid].end()) { // if token already exists
      perThreadLocalEdges[tid][search->second].push_back(std::move(edge));
    } else { // not exist, make a new one
      perThreadTokenToLocalEdgesIdx[tid].insert(
          {token, perThreadLocalEdges[tid].size()});
      std::vector<EdgeDataType> v;
      v.push_back(std::move(edge));
      perThreadLocalEdges[tid].push_back(std::move(v));
    }
  }

  /**
   * Load graph info from the file.
   * Expect a WMD format csv
   *
   * @param filename loaded file for the graph
   *
   */
  void loadGraphFile(const std::string& filename,
                     FileParser<EdgeDataType>& parser,
                     galois::DGAccumulator<uint64_t>& edgeCounter) {
    std::string line;
    struct stat stats;

    std::ifstream graphFile = std::ifstream(filename, std::ifstream::in);
    if (!graphFile.is_open()) {
      printf("cannot open file %s\n", filename.c_str());
      exit(-1);
    }
    stat(filename.c_str(), &stats);

    uint64_t numThreads  = galois::getActiveThreads();
    uint64_t fileSize    = stats.st_size;
    uint64_t bytesPerHost = fileSize / numHosts; 

    uint64_t start     = hostID * bytesPerHost;
    uint64_t end       = start + bytesPerHost;
    // check for partial line at start
    if (hostID != 0) {
      graphFile.seekg(start - 1);
      getline(graphFile, line);
      // if not at start of a line, discard partial line
      if (!line.empty())
        start += line.size();
    }
    // check for partial line at end
    if (hostID != numHosts - 1) {
      graphFile.seekg(end - 1);
      getline(graphFile, line);
      // if not at end of a line, include next line
      if (!line.empty())
        end += line.size();
    } else { // last locale processes to end of file
      end = fileSize;
    }
    graphFile.seekg(start);
    // load segment into memory
    uint64_t segmentLength = end - start;
    char* segmentBuffer    = new char[segmentLength];
    graphFile.read(segmentBuffer, segmentLength);
    if (!graphFile)
      galois::gError("failed to read segment start: ", start, ", end: ", end,
                     ", only ", graphFile.gcount(), " could be read from ",
                     filename);
    galois::gInfo("[", hostID, "] read file, start: ", start, ", end: ", end,
                   "/", fileSize);
    // A parallel loop that parse the segment
    // task 1: get token to global id mapping
    // task 2: get token to edges mapping
    uint64_t lengthPerThread = segmentLength / numThreads;
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      char* currentLine = segmentBuffer + tid * lengthPerThread;
      char* endLine     = currentLine + lengthPerThread;
      // check for partial line
      if (tid != 0) {
        // if not at start of a line, discard partial line
        if (*(currentLine - 1) != '\n')
          currentLine = std::strchr(currentLine, '\n') + 1;
      }
      // last thread processes to end of file
      if (tid == (nthreads - 1))
        endLine = segmentBuffer + segmentLength;
      galois::gDebug("[", hostID, "] thread ", tid,
                     " read file, start: ", currentLine - segmentBuffer,
                     ", end: ", endLine - segmentBuffer, "/", segmentLength);
      uint64_t edgeAdded = 0;
      while (currentLine < endLine) {
        assert(std::strchr(currentLine, '\n'));
        char* nextLine      = std::strchr(currentLine, '\n') + 1;
        uint64_t lineLength = nextLine - currentLine;
        // skip comments
        if (currentLine[0] == '#') {
          currentLine = nextLine;
          continue;
        }
        ParsedGraphStructure<EdgeDataType> value =
            parser.ParseLine(currentLine, lineLength);
        for (auto& edge : value.edges) {
          insertlocalEdgesPerThread(tid, edge.src, edge);
          edgeAdded += 1;
        }
        currentLine = nextLine;
      }
      edgeCounter += edgeAdded;
    });

      delete[] segmentBuffer;
    graphFile.close();
  }

  /**
   * Load graph info from the file.
   * Expect an EdgeList format
   *
   * @param filename loaded file for the graph
   *
   */
  void loadGraphFiles(
      std::vector<std::unique_ptr<FileParser<EdgeDataType>>>&
          parsers) {
    galois::DGAccumulator<uint64_t> edgeCounter;
    edgeCounter.reset();

    // init per thread data struct
    uint64_t numThreads = galois::getActiveThreads();
    perThreadTokenToLocalEdgesIdx.resize(numThreads);
    perThreadLocalEdges.resize(numThreads);

    for (std::unique_ptr<FileParser<EdgeDataType>>& parser :
         parsers) {
        std::string file = parser->GetFiles();
        loadGraphFile(file, *parser, edgeCounter);
    }

    perThreadTokenToLocalEdgesIdx.clear();
    perThreadTokenToLocalEdgesIdx.shrink_to_fit();

    setSizeEdges(edgeCounter.reduce());
  }

  /**
   * Compute global ID of edges by exchange tokenToLocalNodeID
   */
  void buildVtoPHostMap() {
    // determine edgecnt for each host (partial/local)
    std::vector<uint64_t> edgeCnt(numVirtualHosts, 0);
    auto& net               = galois::runtime::getSystemNetworkInterface();
    uint32_t activeThreads  = galois::getActiveThreads();
    uint64_t localEdgesSize = localEdges.size();
    std::vector<std::vector<uint64_t>> threadEdgeCnt(activeThreads);
    for (uint32_t i = 0; i < activeThreads; i++) {
      threadEdgeCnt[i].resize(numVirtualHosts, 0);
    }
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      uint64_t beginNode;
      uint64_t endNode;
      std::tie(beginNode, endNode) =
          galois::block_range((uint64_t)0, localEdgesSize, tid, nthreads);

      for (uint64_t i = beginNode; i < endNode; ++i) {
        uint32_t index = (localEdges[i][0].src) % numVirtualHosts;
        threadEdgeCnt[tid][index] += localEdges[i].size();
      }
    });
    for (uint32_t i = 0; i < activeThreads; i++) {
      for (uint32_t j = 0; j < numVirtualHosts; j++) {
        edgeCnt[j] += threadEdgeCnt[i][j];
      }
    }
    // Send EdgeCnt
    for (unsigned int i = 0; i < numHosts; i++) {
      if (i == hostID)
        continue;

      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, edgeCnt);
      net.sendTagged(i, galois::runtime::evilPhase, std::move(b));
    }
    // Receive edgeCnt
    for (uint32_t h = 0; h < (numHosts - 1); h++) {
      std::vector<uint64_t> recvChunkCounts;

      decltype(net.recieveTagged(galois::runtime::evilPhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase);
      } while (!p);
      galois::runtime::gDeserialize(p->second, recvChunkCounts);
      galois::do_all(galois::iterate((size_t)0, recvChunkCounts.size()),
                     [this, &edgeCnt, &recvChunkCounts](uint64_t i) {
                       edgeCnt[i] += recvChunkCounts[i];
                     });
    }
    increment_evilPhase();

    // Process edgeCnt
    std::vector<uint64_t> edgeCntBkp = edgeCnt;
    uint32_t sf                      = scaleFactor;
    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> cnt_vec;
    for (size_t i = 0; i < edgeCnt.size(); i++) {
      std::vector<uint32_t> vec;
      vec.push_back(i);
      cnt_vec.push_back(std::make_pair(edgeCnt[i], vec));
    }
    std::sort(cnt_vec.begin(), cnt_vec.end());
    while (sf > 1) {
      for (uint32_t i = 0; i < (sf * numHosts / 2); i++) {
        std::pair<uint64_t, std::vector<uint32_t>> mypair;
        cnt_vec[i].first += cnt_vec[sf * numHosts - i - 1].first;
        std::vector vec = cnt_vec[(sf * numHosts) - i - 1].second;
        for (size_t j = 0; j < vec.size(); j++) {
          cnt_vec[i].second.push_back(
              cnt_vec[(sf * numHosts) - i - 1].second[j]);
        }
      }
      sf /= 2;

      std::sort(cnt_vec.begin(), cnt_vec.begin() + (sf * numHosts));
    }
    // Determine virtualToPhyMapping values
    for (uint32_t i = 0; i < numHosts; i++) {
      std::vector vec = cnt_vec[i].second;
      for (size_t j = 0; j < vec.size(); j++) {
        virtualToPhyMapping[vec[j]] = i;
      }
    }
  }

  /**
   * Merge perThread Data Structures
   */
  void mergeThreadDS() {
    // combine per thread edge list
    std::unordered_map<uint64_t, size_t> globalNodeIDToLocalEdgesIdx;
    uint64_t numThreads = perThreadLocalEdges.size();
    for (size_t i = 0; i < numThreads; i++) {
      uint64_t perThreadSize = perThreadLocalEdges[i].size();
      for (size_t j = 0; j < perThreadSize; j++) {
        uint64_t globalID = perThreadLocalEdges[i][j][0].src;
        if (auto search = globalNodeIDToLocalEdgesIdx.find(globalID);
            search != globalNodeIDToLocalEdgesIdx.end()) { // if token already exists
          std::move(perThreadLocalEdges[i][j].begin(),
                    perThreadLocalEdges[i][j].end(),
                    std::back_inserter(localEdges[search->second]));
        } else { // not exist, make a new one
          globalNodeIDToLocalEdgesIdx.insert({globalID, localEdges.size()});
          localEdges.emplace_back(std::move(perThreadLocalEdges[i][j]));
        }
      }
    }
    perThreadLocalEdges.clear();
    perThreadLocalEdges.shrink_to_fit();

  }

public:
  template <typename EdgeListBufferedGraph_EdgeType,
            typename EdgeListBufferedGraph_NodeType>
  friend class EdgeListBufferedGraph;
  std::vector<uint32_t> virtualToPhyMapping;
  uint64_t scaleFactor;
  uint32_t numVirtualHosts;
  std::vector<std::vector<EdgeDataType>>
      localEdges; // edges list of local nodes, idx is local ID

  EdgeListOfflineGraph() {}

  /**
   * An object that loads graph info from the file.
   * Expects an Edgelist format
   *
   * @param name loaded file for the graph.
   * @param md Masters distribution policy that will be used for partition.
   * @param scaleFactor param decide how many virtual host will be used (as a
   * scale of num physical host) Default value is 4. which means there will be 4
   * * numHosts virtual hosts.
   */
  EdgeListOfflineGraph(
      std::vector<std::unique_ptr<
          galois::graphs::FileParser<EdgeDataType>>>& parsers,
      galois::graphs::MASTERS_DISTRIBUTION md,
      uint32_t scaleFactor = 4)
      : OfflineGraph() {
    auto& net         = galois::runtime::getSystemNetworkInterface();
    hostID            = net.ID;
    numHosts          = net.Num;
    this->scaleFactor = scaleFactor;
    numVirtualHosts   = scaleFactor * numHosts;
    virtualToPhyMapping.resize(numVirtualHosts);

    galois::gDebug("[", hostID, "] loadGraphFile!");
    if(md)
      std::cout << "Masters distribution policy is not supported for WMDGraph" << std::endl;
    loadGraphFiles(parsers);
    std::cout << "Graph loaded" << std::endl;
    mergeThreadDS();
    std::cout << "Thread DS merged" << std::endl;
    galois::gDebug("[", hostID, "] Build Virtual To Physical Host Map");
    buildVtoPHostMap();
    std::cout << "VtoP Map built" << std::endl;
  }

  /**
   * Accesses the prefix sum of degree up to node `n`.
   *
   * @param N global ID of node
   * @returns The value located at index n in the edge prefix sum array
   */
  size_t edgeSize() const { return sizeof(EdgeDataType); }

  iterator begin() { return iterator(0); }

  iterator end() { return iterator(size()); }

  ///**
  // * return the end idx of edges of node N
  // *
  // * @param N global ID of node
  // * @return edge_iterator
  // */
  //edge_iterator edge_begin(uint64_t N) {
  //  if (N == 0)
  //    return edge_iterator(0);
  //  else
  //    return edge_iterator(globalEdgePrefixSum[N - 1]);
  //}

  ///**
  // * return the begin idx of edges of node N
  // *
  // * @param N global ID of node
  // * @return edge_iterator
  // */
  //edge_iterator edge_end(uint64_t N) {
  //  return edge_iterator(globalEdgePrefixSum[N]);
  //}

  /**
   * Returns 2 ranges (one for nodes, one for edges) for a particular
   * division. The ranges specify the nodes/edges that a division is
   * responsible for. The function attempts to split them evenly among threads
   * given some kind of weighting
   *
   * @param nodeWeight weight to give to a node in division
   * @param edgeWeight weight to give to an edge in division
   * @param id Division number you want the ranges for
   * @param total Total number of divisions
   * @param scaleFactor Vector specifying if certain divisions should get more
   * than other divisions
   */
  auto divideByNode(size_t nodeWeight, size_t edgeWeight, size_t id,
                    size_t total,
                    std::vector<unsigned> scaleFactor = std::vector<unsigned>())
      -> GraphRange {
    return galois::graphs::divideNodesBinarySearch<EdgeListOfflineGraph>(
        size(), sizeEdges(), nodeWeight, edgeWeight, id, total, *this,
        scaleFactor);
  }
};

/**
 * Class that loads a portion of a Galois graph from disk directly into
 * memory buffers for access.
 *
 * @tparam EdgeDataType type of the edge data
 */
template <typename NodeDataType, typename EdgeDataType>
class EdgeListBufferedGraph : public BufferedGraph<EdgeDataType> {
private:
  typedef boost::counting_iterator<uint64_t> iterator;

  // Edge iterator typedef
  using EdgeIterator = boost::counting_iterator<uint64_t>;

  // specifies whether or not the graph is loaded
  bool graphLoaded = false;

  // size of the entire graph (not just locallly loaded portion)
  uint32_t globalSize = 0;
  // number of edges in the entire graph (not just locallly loaded portion)
  uint64_t globalEdgeSize = 0;

  // number of nodes loaded into this graph
  uint32_t numLocalNodes = 0;
  // number of edges loaded into this graph
  uint64_t numLocalEdges = 0;
  // offset of local to global node id
  uint64_t nodeOffset = 0;

  uint32_t hostID;
  uint32_t numHosts;

  // CSR representation of edges
  std::vector<uint64_t> offsets; // offsets[numLocalNodes] point to end of edges
  std::vector<EdgeDataType> edges;


  /**
   * Exchanges vertex ids to form a global id to local id map before exchanging
   * edges so that using the map edges can be inserted into the edgelist
   */
  void
  gatherEdges(std::vector<std::vector<EdgeDataType>>& localEdges) {
    auto& net              = galois::runtime::getSystemNetworkInterface();
    uint32_t activeThreads = galois::getActiveThreads();

    // prepare both nodedata and edgedata to send to all hosts
    std::vector<std::vector<std::vector<EdgeDataType>>> edgesToSend(
        numHosts, std::vector<std::vector<EdgeDataType>>());
    std::vector<std::vector<uint64_t>> nodesToSend(
        numHosts, std::vector<uint64_t>());

    // PerThread DS
    std::vector<std::vector<std::vector<std::vector<EdgeDataType>>>>
        threadEdgesToSend(
            activeThreads,
            std::vector<std::vector<std::vector<EdgeDataType>>>());
    std::vector<std::vector<std::vector<uint64_t>>> threadNodesToSend(
        activeThreads, std::vector<std::vector<uint64_t>>());
    for (uint32_t i = 0; i < activeThreads; i++) {
      threadEdgesToSend[i].resize(numHosts);
      threadNodesToSend[i].resize(numHosts);
    }

    // Prepare edgeList and Vertex ID list to send to other hosts
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      uint64_t beginNode;
      uint64_t endNode;
      std::tie(beginNode, endNode) =
          galois::block_range((uint64_t)0, localEdges.size(), tid, nthreads);

      for (uint64_t i = beginNode; i < endNode; ++i) {
        uint64_t src = localEdges[i][0].src;
        int host = virtualToPhyMapping[src % numVirtualHosts];
        threadEdgesToSend[tid][host].push_back((localEdges[i]));
      }
    });

    // Prepare Nodedata to send to other hosts
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      size_t beginNode;
      size_t endNode;
      std::tie(beginNode, endNode) =
          galois::block_range((uint64_t)0, localEdges.size(), tid, nthreads);
      for (size_t i = beginNode; i < (endNode); ++i) {
        int host =
            virtualToPhyMapping[(localEdges[i][0].src) % (scaleFactor * numHosts)];
        threadNodesToSend[tid][host].push_back((localEdges[i][0].src));
      }
    });
    for (uint32_t tid = 0; tid < activeThreads; tid++) {
      for (uint32_t h = 0; h < numHosts; h++) {
        nodesToSend[h].insert(nodesToSend[h].end(),
                              threadNodesToSend[tid][h].begin(),
                              threadNodesToSend[tid][h].end());
        edgesToSend[h].insert(edgesToSend[h].end(),
                              threadEdgesToSend[tid][h].begin(),
                              threadEdgesToSend[tid][h].end());
      }
    }
    threadNodesToSend.clear();
    threadEdgesToSend.clear();
    localEdges.clear();
    // Send Nodelist
    for (uint32_t h = 0; h < numHosts; h++) {
      if (h == hostID)
        continue;
      galois::runtime::SendBuffer sendBuffer;
      galois::runtime::gSerialize(sendBuffer, nodesToSend[h]);
      net.sendTagged(h, galois::runtime::evilPhase, std::move(sendBuffer));
    }
    // Collect node data received from other hosts
    for (uint32_t i = 0; i < (numHosts - 1); i++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase);
      } while (!p);
      std::vector<uint64_t> NodeData;
      galois::runtime::gDeserialize(p->second, NodeData);
      std::vector<std::map<uint64_t, uint32_t>> threadMap(activeThreads);
      std::vector<std::map<uint32_t, uint64_t>> threadLIDMap(activeThreads);
      uint64_t offset = GIDtoLID.size();
      galois::on_each([&](unsigned tid, unsigned nthreads) {
        size_t beginNode;
        size_t endNode;
        std::tie(beginNode, endNode) =
            galois::block_range((uint64_t)0, NodeData.size(), tid, nthreads);
        uint64_t delta;
        delta = std::ceil((double)NodeData.size() / activeThreads);
        uint64_t cnt = 0;
        for (size_t j = beginNode; j < (endNode); ++j) {
          if(GIDtoLID.find(NodeData[j]) != GIDtoLID.end())
            continue;
          threadMap[tid][NodeData[j]] =
              offset + (tid * (delta)) + cnt - beginNode;
          threadLIDMap[tid][offset + (tid * (delta)) + cnt - beginNode] = 
              NodeData[j];
          cnt++;
        }
      });
      for (uint32_t t = 0; t < activeThreads; t++) {
        GIDtoLID.insert(threadMap[t].begin(), threadMap[t].end());
        LIDtoGID.insert(threadLIDMap[t].begin(), threadLIDMap[t].end());
      }
      threadMap.clear();
      NodeData.clear();
      threadLIDMap.clear();
    }

    // Collect node data present in this host
    std::vector<std::map<uint64_t, size_t>> threadMap(activeThreads);
    std::vector<std::map<size_t, uint64_t>> threadLIDMap(activeThreads);
    uint64_t offset = GIDtoLID.size();
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      size_t beginNode;
      size_t endNode;
      std::tie(beginNode, endNode) = galois::block_range(
          (uint64_t)0, nodesToSend[hostID].size(), tid, nthreads);
      uint64_t delta;
      delta = std::ceil((double)nodesToSend[hostID].size() / activeThreads);
      uint64_t cnt = 0;
      for (size_t i = beginNode; i < (endNode); ++i) {
          if(GIDtoLID.find(nodesToSend[hostID][i]) != GIDtoLID.end())
            continue;
        threadMap[tid][nodesToSend[hostID][i]] =
            offset + (tid * (delta)) + cnt - beginNode;
        threadLIDMap[tid][offset + (tid * (delta)) + cnt - beginNode] = 
            nodesToSend[hostID][i];
        cnt++;
      }
    });
    for (uint32_t t = 0; t < activeThreads; t++) {
      GIDtoLID.insert(threadMap[t].begin(), threadMap[t].end());
      LIDtoGID.insert(threadLIDMap[t].begin(), threadLIDMap[t].end());
    }
    threadMap.clear();
    threadLIDMap.clear();
    numLocalNodes = GIDtoLID.size();
    localEdges.clear();
    nodesToSend.clear();


    increment_evilPhase();
    // Send Edgelist
    for (uint32_t h = 0; h < numHosts; h++) {
      if (h == hostID)
        continue;
      galois::runtime::SendBuffer sendBuffer;
      galois::runtime::gSerialize(sendBuffer, edgesToSend[h]);
      galois::gInfo("[", hostID, "] ", "send to ", h,
                    " edgesToSend size: ", edgesToSend[h].size());
      net.sendTagged(h, galois::runtime::evilPhase, std::move(sendBuffer));
    }
    // Appending edges in each host that belong to self
    localEdges.resize(GIDtoLID.size());

    // Receiving edges from other hosts and populating edgelist
    for (uint32_t h = 0; h < (numHosts - 1); h++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase);
      } while (!p);
      uint32_t sendingHost = p->first;
      std::vector<std::vector<EdgeDataType>> edgeList;
      galois::runtime::gDeserialize(p->second, edgeList);
      galois::gInfo("[", hostID, "] recv from ", sendingHost,
                    " edgeList size: ", edgeList.size());
      galois::on_each([&](unsigned tid, unsigned nthreads) {
        size_t beginNode;
        size_t endNode;
        std::tie(beginNode, endNode) =
            galois::block_range((size_t)0, edgeList.size(), tid, nthreads);
        for (size_t j = beginNode; j < endNode; j++) {
          assert(GIDtoLID.find(edgeList[j][0].src) != GIDtoLID.end());
          auto lid = GIDtoLID[edgeList[j][0].src];
          localEdges[lid].insert(
              std::end(localEdges[lid]),
              std::begin(edgeList[j]), std::end(edgeList[j]));
        }
      });
      edgeList.clear();
    }
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      size_t beginNode;
      size_t endNode;
      std::tie(beginNode, endNode) = galois::block_range(
          (size_t)0, edgesToSend[hostID].size(), tid, nthreads);
      for (size_t j = beginNode; j < endNode; j++) {
        auto lid = GIDtoLID[edgesToSend[hostID][j][0].src];
        localEdges[lid].insert(
            std::end(localEdges[lid]),
            std::begin(edgesToSend[hostID][j]),
            std::end(edgesToSend[hostID][j]));
      }
    });
    increment_evilPhase();
    edgesToSend.clear();
    localNodeSize[hostID] = localEdges.size();
  }

  void exchangeNodeData(EdgeListOfflineGraph<galois::Vertex, galois::Edge>& srcGraph) {

    auto& net              = galois::runtime::getSystemNetworkInterface();
    // send vertex size to other hosts
    for (uint32_t h = 0; h < numHosts; ++h) {
      if (h == hostID) {
        continue;
      }
      // serialize size_t
      galois::runtime::SendBuffer sendBuffer;
      galois::runtime::gSerialize(sendBuffer, localNodeSize);
      net.sendTagged(h, galois::runtime::evilPhase, std::move(sendBuffer));
    }

    for (uint32_t h = 0; h < numHosts - 1; h++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase);
      } while (!p);
      std::vector<uint64_t> cnt;
      // deserialize local_node_size
      galois::runtime::gDeserialize(p->second, cnt);
      assert(cnt.size() == numHosts);
      for (uint32_t i = 0; i < numHosts; i++) {
        localNodeSize[i] += cnt[i];
      }
    }

    numNodes      = localNodeSize[hostID];
    numLocalNodes = numNodes;
    // compute prefix sum to get offset
    for (size_t h = 1; h < numHosts; h++) {
      globalNodeOffset[h] = localNodeSize[h - 1] + globalNodeOffset[h - 1];
    }
    srcGraph.setSize(globalNodeOffset[numHosts - 1] +
                     localNodeSize[numHosts - 1]);
    increment_evilPhase();
  }

  /**
   * Flatten the 2D vector localEdges into a CSR edge list
   * Will compute edge size and build CSR edge offset mapping
   */
  void flattenEdges(std::vector<std::vector<EdgeDataType>>& localEdges) {
    // build CSR edge offsets
    offsets.resize(numLocalNodes + 1, 0);
    for (size_t i = 0; i < numLocalNodes; i++) {
      uint64_t cnt;
      if (i >= localEdges.size())
        cnt = 0;
      else
        cnt = localEdges[i].size();
      offsets[i + 1] += cnt + offsets[i];
    }
    numLocalEdges = offsets[numLocalNodes];

    // build flatten edge list
    edges.resize(numLocalEdges);
    galois::do_all(
        galois::iterate((size_t)0, localEdges.size()),
        [this, &localEdges](size_t i) {
          std::move(localEdges[i].begin(), localEdges[i].end(),
                    edges.begin() + offsets[i]);
        },
        galois::steal());
  }

public:
  EdgeListBufferedGraph() : BufferedGraph<EdgeDataType>() {}
  std::unordered_map<uint64_t, uint32_t> GIDtoLID;
  std::unordered_map<uint32_t, uint64_t> LIDtoGID;

  // copy not allowed
  //! disabled copy constructor
  EdgeListBufferedGraph(const EdgeListBufferedGraph&) = delete;
  //! disabled copy constructor operator
  EdgeListBufferedGraph& operator=(const EdgeListBufferedGraph&) = delete;
  // move not allowed
  //! disabled move operator
  EdgeListBufferedGraph(EdgeListBufferedGraph&&) = delete;
  //! disabled move constructor operator
  EdgeListBufferedGraph& operator=(EdgeListBufferedGraph&&) = delete;

  uint32_t scaleFactor;
  uint64_t numNodes;
  uint32_t numVirtualHosts;
  std::vector<uint64_t> localNodeSize; // number of local nodes in each hosts
  std::vector<uint64_t>
      globalNodeOffset; // each hosts' local ID offset wrt global ID
  std::vector<uint32_t> virtualToPhyMapping;
  /**
   * Gets the number of global nodes in the graph
   * @returns the total number of nodes in the graph (not just local loaded
   * nodes)
   */
  uint32_t size() const { return globalSize; }

  /**
   * Gets the number of global edges in the graph
   * @returns the total number of edges in the graph (not just local loaded
   * edges)
   */
  uint32_t sizeEdges() const { return globalEdgeSize; }

  /**
   * Gets the number of local edges in the graph
   * @returns the total number of edges in the local graph
   */
  uint32_t sizeLocalEdges() const { return numLocalEdges; }

  //! @returns node offset of this buffered graph
  uint64_t getNodeOffset() const { return nodeOffset; }

  /**
   * Given a node/edge range to load, loads the specified portion of the
   * graph into memory buffers from OfflineGraph.
   *
   * @param srcGraph the OfflineGraph to load from
   * @param numGlobalEdges Total number of edges in the graph
   */
  void loadPartialGraph(EdgeListOfflineGraph<galois::Vertex, galois::Edge>& srcGraph,
                        uint64_t numGlobalEdges) {
    if (graphLoaded) {
      GALOIS_DIE("Cannot load an buffered graph more than once.");
    }

    // prepare meta data
    auto& net = galois::runtime::getSystemNetworkInterface();
    hostID    = net.ID;
    numHosts  = net.Num;

    globalEdgeSize = numGlobalEdges;

    scaleFactor     = srcGraph.scaleFactor;
    numVirtualHosts = srcGraph.numVirtualHosts;
    virtualToPhyMapping.resize(numVirtualHosts);
    for (uint32_t i = 0; i < numVirtualHosts; i++) {
      virtualToPhyMapping[i] = srcGraph.virtualToPhyMapping[i];
    }

    // build local buffered graph 
    globalNodeOffset.assign(numHosts, 0);
    localNodeSize.resize(numHosts, 0);
    galois::gDebug("[", hostID, "] gatherVerticesAndEdges!");
    gatherEdges(srcGraph.localEdges);
    galois::gDebug("[", hostID, "] Exchange Node Info!");
    exchangeNodeData(srcGraph);
    galois::gInfo("[", hostID, "] ", "flattenEdges!");
    flattenEdges(srcGraph.localEdges);
    // clean unused data
    srcGraph.localEdges.clear();
    srcGraph.localEdges.shrink_to_fit();
    graphLoaded = true;
    galois::gInfo("[", hostID, "] ", "exchangeNodeRange!");
    galois::gDebug("[", hostID, "] ",
                   "BufferedGraph built, nodes: ", numLocalNodes,
                   ", edges: ", numLocalEdges);
  }

  /**
   * Gather local nodes data (mirror + master nodes) from other hosts to
   * this host And save data to graph
   *
   * @param srcGraph the OfflineGraph owns node data (will be cleared after
   * this call)
   * @param proxiesOnHosts a list of bit vector which indicates node on that
   * hosts (include mirror and master nodes)
   * @param totalLocalNodes the total number of local nodes this host should
   * have (include mirror and master nodes)
   */

  /**
   * Get the index to the first edge of the provided node THAT THIS GRAPH
   * HAS LOADED (not necessary the first edge of it globally).
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @returns a LOCAL edge id iterator
   */
  EdgeIterator edgeBegin(uint64_t globalNodeID) {
    assert((globalNodeID - globalNodeOffset[hostID]) < GIDtoLID.size());
    return EdgeIterator(offsets[globalNodeID - globalNodeOffset[hostID]]);
  }

  /**
   * Get the index to the first edge of the node after the provided node.
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @returns a LOCAL edge id iterator
   */
  EdgeIterator edgeEnd(uint64_t globalNodeID) {
    assert((globalNodeID - globalNodeOffset[hostID]) < GIDtoLID.size());
    return EdgeIterator(offsets[globalNodeID + 1 - globalNodeOffset[hostID]]);
  }

  /**
   * Get the global node id of the destination of the provided edge.
   *
   * @param localEdgeID the local edge id of the edge to get the destination
   * for (should obtain from edgeBegin/End)
   */
  uint64_t edgeDestination(uint64_t localEdgeID) {
    return edges[localEdgeID].dst;
  }

  /**
   * Get the edge data of some edge.
   *
   * @param localEdgeID the local edge id of the edge to get the data of
   * @returns the edge data of the requested edge id
   */
  template <typename K = EdgeDataType,
            typename std::enable_if<!std::is_void<K>::value>::type* = nullptr>
  EdgeDataType edgeData(uint64_t localEdgeID) {
    assert(localEdgeID < numLocalEdges);
    return edges[localEdgeID];
  }

  /**
   * Version of above function when edge data type is void.
   */
  template <typename K = EdgeDataType,
            typename std::enable_if<std::is_void<K>::value>::type* = nullptr>
  unsigned edgeData(uint64_t) {
    galois::gWarn("Getting edge data on graph when it doesn't exist\n");
    return 0;
  }

  /**
   * Get the number of edges of the node
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @returns number of edges
   */
  uint64_t edgeNum(uint64_t globalNodeID) {
    return offsets[globalNodeID - globalNodeOffset[hostID] + 1] - offsets[globalNodeID - globalNodeOffset[hostID]];
  }

  /**
   * Get the dst of edges of the node
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @param G2L the global to local id mapping
   * @returns a vector of dst local node id
   */
  std::vector<uint64_t> edgeLocalDst(uint64_t globalNodeID) {
    std::vector<uint64_t> dst;
    uint64_t end = offsets[globalNodeID - globalNodeOffset[hostID] + 1];
    for (auto itr = offsets[globalNodeID - globalNodeOffset[hostID]]; itr != end; ++itr) {
        dst.push_back(edges[itr].dst); 
    }
    return dst;
  }

  /**
   * Get the data of edges of the node
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @returns a pointer to the first edges of the node in the buffer
   */
  EdgeDataType* edgeDataPtr(uint64_t globalNodeID) {
    return edges.data() + offsets[globalNodeID - globalNodeOffset[hostID]];
  }

  /**
   * Free all of the in memory buffers in this object.
   */
  void resetAndFree() {
    offsets.clear();
    offsets.shrink_to_fit();
    edges.clear();
    edges.shrink_to_fit();
  }
};
} // namespace graphs
} // namespace galois
#endif
