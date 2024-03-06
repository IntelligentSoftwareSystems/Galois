/**
 * @file WMDGraph.h
 *
 * Contains the implementation of WMDBufferedGraph and WMDOfflineGraph which is
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

#include "graphTypes.h"
#include "data_types.h"
#include "graph.h"
#include "schema.h"
#include "instrument.h"

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
 * Load a WMD format graph from file into memory.
 *
 * Inherit from OffilineGraph only to make it compatible with Partitioner
 * Internal implementation are completed different.
 */
template <typename NodeDataType, typename EdgeDataType>
class WMDOfflineGraph : public OfflineGraph {
protected:
  // TODO: consider typedef uint64_t NodeIDType ?
  typedef boost::counting_iterator<uint64_t> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;

  // private feilds from base class that will be updated
  // uint64_t numNodes;  // num of global nodes
  // uint64_t numEdges;  // num of global edges

  // local feilds (different on each hosts)
  std::vector<uint64_t> localNodeSize; // number of local nodes in each hosts
  uint64_t localEdgeSize;              // number of local edges in this host

  // TODO: it may be possible to optimize these vectors by numa aware data
  // structures
  std::vector<uint64_t>
      localEdgesIdxToGlobalNodeID; // map idx in localEdges to global node ID
  std::vector<NodeDataType> localNodes; // nodes in this host, index by local ID

  // global feilds (same on each hosts)
  std::vector<uint64_t> nodeOffset; // each hosts' local ID offset wrt global ID
  std::vector<uint64_t>
      globalEdgePrefixSum; // a prefix sum of degree of each global nodes

  // per thread data struct (will be combined into a single data struct)
  std::vector<std::unordered_map<uint64_t, size_t>>
      perThreadTokenToLocalEdgesIdx;
  std::vector<std::vector<NodeDataType>> perThreadLocalNodes;
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

  inline void insertlocalEdgesPerThread(unsigned tid, uint64_t token,
                                        EdgeDataType& edge) {
    I_RR();
    if (auto search = perThreadTokenToLocalEdgesIdx[tid].find(token);
        search !=
        perThreadTokenToLocalEdgesIdx[tid].end()) { // if token already exists
      I_WR();
      perThreadLocalEdges[tid][search->second].push_back(std::move(edge));
    } else { // not exist, make a new one
      I_WR();
      perThreadTokenToLocalEdgesIdx[tid].insert(
          {token, perThreadLocalEdges[tid].size()});
      I_WR();
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
   * @param segmentsPerHost the number of file segments each host will load.
   * If value is 1, no file striping is performed. The file is striped into
   * (segementsPerHost * numHosts) segments.
   * @param setEdgeSize if True, will update local edges size on this step.
   * Only set to ture when prefixsum will not be computed.
   *
   * @details File striping is used to randomize the order of nodes/edges
   * loaded from the graph. WMD dataset csv typically grouped nodes/edges by its
   * types, which will produce an imbalanced graph if you break the file evenly
   * among hosts. So file striping make each host be able to load multiple
   * segments in different positions of the file, which produced a more balanced
   * graph.
   */
  void loadGraphFile(const std::string& filename,
                     FileParser<NodeDataType, EdgeDataType>& parser,
                     uint64_t segmentsPerHost,
                     galois::GAccumulator<uint64_t>& nodeCounter,
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
    uint64_t numSegments = numHosts * segmentsPerHost;
    uint64_t fileSize    = stats.st_size;
    uint64_t bytesPerSegment =
        fileSize / numSegments; // file size / number of segments

    // for each host N, it will read segment like:
    // N, N + numHosts, N + numHosts * 2, ..., N + numHosts * (segmentsPerHost -
    // 1)
    for (uint64_t cur = 0; cur < segmentsPerHost; cur++) {
      uint64_t segmentID = hostID + cur * numHosts;
      uint64_t start     = segmentID * bytesPerSegment;
      uint64_t end       = start + bytesPerSegment;

      // check for partial line at start
      if (segmentID != 0) {
        graphFile.seekg(start - 1);
        getline(graphFile, line);
        I_RS();

        // if not at start of a line, discard partial line
        if (!line.empty())
          start += line.size();
      }

      // check for partial line at end
      if (segmentID != numSegments - 1) {
        graphFile.seekg(end - 1);
        getline(graphFile, line);
        I_RS();

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
      I_RS();

      if (!graphFile)
        galois::gError("failed to read segment start: ", start, ", end: ", end,
                       ", only ", graphFile.gcount(), " could be read from ",
                       filename);
      galois::gDebug("[", hostID, "] read file, start: ", start, ", end: ", end,
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
          I_RR();
        }

        // last thread processes to end of file
        if (tid == (nthreads - 1))
          endLine = segmentBuffer + segmentLength;
        galois::gDebug("[", hostID, "] thread ", tid,
                       " read file, start: ", currentLine - segmentBuffer,
                       ", end: ", endLine - segmentBuffer, "/", segmentLength);
        // init per thread counter
        uint64_t edgeAdded = 0;
        while (currentLine < endLine) {
          assert(std::strchr(currentLine, '\n'));
          char* nextLine      = std::strchr(currentLine, '\n') + 1;
          uint64_t lineLength = nextLine - currentLine;
          I_RR();

          // skip comments
          if (currentLine[0] == '#') {
            currentLine = nextLine;
            continue;
          }

          // delimiter and # tokens set for wmd data file
          ParsedGraphStructure<NodeDataType, EdgeDataType> value =
              parser.ParseLine(currentLine, lineLength);
          I_RS();

          if (value.isNode) {
            I_WR();
            perThreadLocalNodes[tid].emplace_back(value.node);
          } else if (value.isEdge) {
            for (auto& edge : value.edges) {
              insertlocalEdgesPerThread(tid, edge.src, edge);
              edgeAdded += 1;
            }
          }
          currentLine = nextLine;
        }
        // update accumulator
        edgeCounter += edgeAdded;
        if (cur == segmentsPerHost - 1) {
          nodeCounter += perThreadLocalNodes[tid].size();
          I_RR();
        }
      });

      delete[] segmentBuffer;
    }
    graphFile.close();
  }

  /**
   * Load graph info from the file.
   * Expect a WMD format csv
   *
   * @param filename loaded file for the graph
   * @param segmentsPerHost the number of file segments each host will load.
   * If value is 1, no file striping is performed. The file is striped into
   * (segementsPerHost * numHosts) segments.
   * @param setEdgeSize if True, will update local edges size on this step.
   * Only set to ture when prefixsum will not be computed.
   *
   * @details File striping is used to randomize the order of nodes/edges
   * loaded from the graph. WMD dataset csv typically grouped nodes/edges by its
   * types, which will produce an imbalanced graph if you break the file evenly
   * among hosts. So file striping make each host be able to load multiple
   * segments in different positions of the file, which produced a more balanced
   * graph.
   */
  void loadGraphFiles(
      std::vector<std::unique_ptr<FileParser<NodeDataType, EdgeDataType>>>&
          parsers,
      uint64_t segmentsPerHost, bool setEdgeSize) {
    galois::GAccumulator<uint64_t> nodeCounter;
    nodeCounter.reset();
    galois::DGAccumulator<uint64_t> edgeCounter;
    edgeCounter.reset();

    // init per thread data struct
    uint64_t numThreads = galois::getActiveThreads();
    perThreadTokenToLocalEdgesIdx.resize(numThreads);
    perThreadLocalNodes.resize(numThreads);
    perThreadLocalEdges.resize(numThreads);

    for (std::unique_ptr<FileParser<NodeDataType, EdgeDataType>>& parser :
         parsers) {
      I_RR();
      for (const std::string& file : parser->GetFiles()) {
        I_RR();
        loadGraphFile(file, *parser, segmentsPerHost, nodeCounter, edgeCounter);
      }
    }

    perThreadTokenToLocalEdgesIdx.clear();
    perThreadTokenToLocalEdgesIdx.shrink_to_fit();

    if (setEdgeSize) {
      setSizeEdges(edgeCounter.reduce());
    }

    I_RR();
    localEdgeSize = edgeCounter.read_local();
    localNodeSize.resize(numHosts);
    I_BM(numHosts);
    localNodeSize[hostID] = nodeCounter.reduce();
  }

  /**
   * Compute global ID of edges by exchange tokenToLocalNodeID
   */
  void exchangeEdgeCnt() {
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
        I_WR();
      }
    });
    for (uint32_t i = 0; i < activeThreads; i++) {
      for (uint32_t j = 0; j < numVirtualHosts; j++) {
        edgeCnt[j] += threadEdgeCnt[i][j];
        I_RS();
        I_WR();
      }
    }
    // Send EdgeCnt
    for (unsigned int i = 0; i < numHosts; i++) {
      if (i == hostID)
        continue;

      I_WM(edgeCnt.size());
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, edgeCnt);
      net.sendTagged(i, galois::runtime::evilPhase, b);
    }
    // Receive edgeCnt
    for (uint32_t h = 0; h < (numHosts - 1); h++) {
      std::vector<uint64_t> recvChunkCounts;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      galois::runtime::gDeserialize(p->second, recvChunkCounts);
      I_LC(p->first, recvChunkCounts.size() * sizeof(uint64_t));
      galois::do_all(galois::iterate((size_t)0, recvChunkCounts.size()),
                     [this, &edgeCnt, &recvChunkCounts](uint64_t i) {
                       I_RR();
                       I_WR();
                       edgeCnt[i] += recvChunkCounts[i];
                     });
    }
    increment_evilPhase();
    uint64_t edgesNum = 0;
    for (uint32_t h = 0; h < numVirtualHosts; h++) {
      I_RS();
      edgesNum += edgeCnt[h];
    }
    setSizeEdges(edgesNum);
    I_WR();
    // Process edgeCnt
    std::vector<uint64_t> edgeCntBkp = edgeCnt;
    uint32_t sf                      = scaleFactor;
    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> cnt_vec;
    for (size_t i = 0; i < edgeCnt.size(); i++) {
      std::vector<uint32_t> vec;
      vec.push_back(i);
      cnt_vec.push_back(std::make_pair(edgeCnt[i], vec));
      I_WR();
      I_RS();
    }
    std::sort(cnt_vec.begin(), cnt_vec.end());
    while (sf > 1) {
      for (uint32_t i = 0; i < (sf * numHosts / 2); i++) {
        std::pair<uint64_t, std::vector<uint32_t>> mypair;
        I_RR();
        I_RR();
        I_WR();
        cnt_vec[i].first += cnt_vec[sf * numHosts - i - 1].first;
        std::vector vec = cnt_vec[(sf * numHosts) - i - 1].second;
        for (size_t j = 0; j < vec.size(); j++) {
          cnt_vec[i].second.push_back(
              cnt_vec[(sf * numHosts) - i - 1].second[j]);
          I_RS();
          I_WR();
        }
      }
      sf /= 2;

#ifdef GALOIS_INSTRUMENT
      std::sort(cnt_vec.begin(), cnt_vec.begin() + (sf * numHosts));
      for (uint32_t i = 0; i < (uint32_t)sf * numHosts; i++) {
        I_RR();
        I_WR();
      }
#endif
    }
    // Determine virtualToPhyMapping values
    for (uint32_t i = 0; i < numHosts; i++) {
      std::vector vec = cnt_vec[i].second;
      for (size_t j = 0; j < vec.size(); j++) {
        virtualToPhyMapping[vec[j]] = i;
        I_RS();
        I_WR();
      }
    }
  }

  /**
   * Merge perThread Data Structures
   */
  void mergeThreadDS() {
    // combine per thread edge list
    // TODO: It may cause memory fragmentation and so use vector +
    // inspector/executor in that case
    std::unordered_map<uint64_t, size_t> globalNodeIDToLocalEdgesIdx;
    uint64_t numThreads = perThreadLocalEdges.size();
    for (size_t i = 0; i < numThreads; i++) {
      I_RR();
      uint64_t perThreadSize = perThreadLocalEdges[i].size();
      for (size_t j = 0; j < perThreadSize; j++) {
        I_RR();
        uint64_t globalID = perThreadLocalEdges[i][j][0].src;
        I_RR();
        if (auto search = globalNodeIDToLocalEdgesIdx.find(globalID);
            search !=
            globalNodeIDToLocalEdgesIdx.end()) { // if token already exists
          I_WM(perThreadLocalEdges[i][j].size());
          std::move(perThreadLocalEdges[i][j].begin(),
                    perThreadLocalEdges[i][j].end(),
                    std::back_inserter(localEdges[search->second]));
        } else { // not exist, make a new one
          I_WR();
          globalNodeIDToLocalEdgesIdx.insert({globalID, localEdges.size()});
          I_WR();
          localEdges.emplace_back(std::move(perThreadLocalEdges[i][j]));
        }
      }
    }
    perThreadLocalEdges.clear();
    perThreadLocalEdges.shrink_to_fit();

    // make a maping from localEdges idx to ID
    localEdgesIdxToGlobalNodeID.resize(globalNodeIDToLocalEdgesIdx.size());
    galois::do_all(
        galois::iterate(globalNodeIDToLocalEdgesIdx),
        [this](std::unordered_map<uint64_t, size_t>::value_type& p) {
          I_WR();
          localEdgesIdxToGlobalNodeID[p.second] = p.first;
        },
        galois::steal());

    // combine per thread node list
    std::vector<uint64_t> perThreadLocalNodesOffset(perThreadLocalNodes.size(),
                                                    0);
    for (size_t i = 1; i < perThreadLocalNodes.size(); i++) {
      I_WR();
      perThreadLocalNodesOffset[i] =
          perThreadLocalNodes[i - 1].size() + perThreadLocalNodesOffset[i - 1];
    }
    localNodes.resize(localNodeSize[hostID]);
    galois::on_each([&](unsigned tid, unsigned) {
      uint64_t perThreadOffset = perThreadLocalNodesOffset[tid];
      I_WM(perThreadLocalNodes[tid].size());
      std::move(perThreadLocalNodes[tid].begin(),
                perThreadLocalNodes[tid].end(),
                localNodes.begin() + perThreadOffset);
    });
    perThreadLocalNodes.clear();
    perThreadLocalNodes.shrink_to_fit();
  }

  /**
   * Compute prefix sum of the size of edges of nodes in the graph
   */
  void computeEdgePrefixSum() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    size_t numLocalNodes    = localEdges.size();
    uint64_t numGlobalNodes = size();
    std::vector<uint64_t> localNodeDegree(numLocalNodes);

    galois::do_all(
        galois::iterate((size_t)0, numLocalNodes),
        [this, &localNodeDegree](size_t n) {
          I_WR();
          localNodeDegree[n] = localEdges[n].size();
        },
        galois::steal());

    // broadcast node degrees and its global ID to other hosts
    {
      galois::runtime::SendBuffer sendBuffer;
      galois::runtime::gSerialize(sendBuffer, localNodeDegree);
      galois::runtime::gSerialize(
          sendBuffer,
          localEdgesIdxToGlobalNodeID); // global ID of the localNodeDegree

      for (uint32_t h = 0; h < numHosts; ++h) {
        if (h == hostID) {
          continue;
        }

        galois::runtime::SendBuffer b;
        galois::runtime::gSerialize(b, sendBuffer);
        I_LC(h, b.size());
        net.sendTagged(h, galois::runtime::evilPhase, b);
      }
    }

    // init edge prefix sum
    globalEdgePrefixSum.resize(numGlobalNodes);
    galois::do_all(
        galois::iterate((size_t)0, localEdgesIdxToGlobalNodeID.size()),
        [this, &localNodeDegree](size_t n) {
          I_WR();
          globalEdgePrefixSum[localEdgesIdxToGlobalNodeID[n]] +=
              localNodeDegree[n];
        },
        galois::steal());
    localNodeDegree.clear();
    localNodeDegree.shrink_to_fit();

    // recv node degrees and its global ID from other hosts
    // build a list of degree of all global nodes on `globalEdgePrefixSum`
    for (uint32_t h = 0; h < numHosts - 1; h++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      // deserialize
      std::vector<uint64_t> recvNodeDegree;
      std::vector<uint64_t> recvNodeGlobalID;
      galois::runtime::gDeserialize(p->second, recvNodeDegree);
      galois::runtime::gDeserialize(p->second, recvNodeGlobalID);

      galois::do_all(
          galois::iterate((size_t)0, recvNodeDegree.size()),
          [this, &recvNodeDegree, &recvNodeGlobalID](size_t n) {
            I_WR();
            globalEdgePrefixSum[recvNodeGlobalID[n]] += recvNodeDegree[n];
          },
          galois::steal());
    }

    // globalEdgePrefixSum has degree info now, so could compute prefixsum
    // in place
    for (size_t h = 1; h < numGlobalNodes; h++) {
      I_WR();
      globalEdgePrefixSum[h] += globalEdgePrefixSum[h - 1];
    }

    // set numEdges (global size)
    setSizeEdges(globalEdgePrefixSum[numGlobalNodes - 1]);
    increment_evilPhase();
  }

public:
  template <typename WMDBufferedGraph_EdgeType,
            typename WMDBufferedGraph_NodeType>
  friend class WMDBufferedGraph;
  std::vector<uint32_t> virtualToPhyMapping;
  uint64_t scaleFactor;
  uint32_t numVirtualHosts;
  std::vector<std::vector<EdgeDataType>>
      localEdges; // edges list of local nodes, idx is local ID

  WMDOfflineGraph() {}

  /**
   * An object that load graph info from the file.
   * Expect a WMD format csv
   *
   * @param name loaded file for the graph.
   * @param md Masters distribution policy that will be used for partition.
   * @param segmentsPerHost the number of file segments each host will load.
   * Default value is 1, no file striping is performed. The file is striped into
   * (segementsPerHost * numHosts) segments.
   * @param scaleFactor param decide how many virtual host will be used (as a
   * scale of num physical host) Default value is 4. which means there will be 4
   * * numHosts virtual hosts.
   */
  WMDOfflineGraph(
      std::vector<std::unique_ptr<
          galois::graphs::FileParser<NodeDataType, EdgeDataType>>>& parsers,
      galois::graphs::MASTERS_DISTRIBUTION md, uint64_t segmentsPerHost = 1,
      uint32_t scaleFactor = 4)
      : OfflineGraph() {
    auto& net         = galois::runtime::getSystemNetworkInterface();
    hostID            = net.ID;
    numHosts          = net.Num;
    this->scaleFactor = scaleFactor;
    numVirtualHosts   = scaleFactor * numHosts;
    virtualToPhyMapping.resize(numVirtualHosts);

    galois::gDebug("[", hostID, "] loadGraphFile!");
    loadGraphFiles(parsers, segmentsPerHost, md == BALANCED_MASTERS);
    mergeThreadDS();
    galois::gDebug("[", hostID, "] exchangeEdgeCntMetadata!");
    exchangeEdgeCnt();
    galois::gInfo("[", hostID, "] read WMD csv file with local Nodes: ",
                  localNodeSize[hostID], ", local Edges: ", localEdgeSize);
  }

  /**
   * Accesses the prefix sum of degree up to node `n`.
   *
   * @param N global ID of node
   * @returns The value located at index n in the edge prefix sum array
   */
  uint64_t operator[](uint64_t N) { return globalEdgePrefixSum[N]; }

  size_t edgeSize() const { return sizeof(EdgeDataType); }

  iterator begin() { return iterator(0); }

  iterator end() { return iterator(size()); }

  /**
   * return the end idx of edges of node N
   *
   * @param N global ID of node
   * @return edge_iterator
   */
  edge_iterator edge_begin(uint64_t N) {
    if (N == 0)
      return edge_iterator(0);
    else
      return edge_iterator(globalEdgePrefixSum[N - 1]);
  }

  /**
   * return the begin idx of edges of node N
   *
   * @param N global ID of node
   * @return edge_iterator
   */
  edge_iterator edge_end(uint64_t N) {
    return edge_iterator(globalEdgePrefixSum[N]);
  }

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
    return galois::graphs::divideNodesBinarySearch<WMDOfflineGraph>(
        size(), sizeEdges(), nodeWeight, edgeWeight, id, total, *this,
        scaleFactor);
  }

  /**
   * Release memory used by EdgePrefixSum
   * After that, calls to `edge_begin` and `edge_end` will be invalid
   */
  void clearEdgePrefixSumInfo() {
    globalEdgePrefixSum.clear();
    globalEdgePrefixSum.shrink_to_fit();
  }
};

/**
 * Class that loads a portion of a Galois graph from disk directly into
 * memory buffers for access.
 *
 * @tparam EdgeDataType type of the edge data
 */
template <typename NodeDataType, typename EdgeDataType>
class WMDBufferedGraph : public BufferedGraph<EdgeDataType> {
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

  void
  exchangeLocalNodeSize(WMDOfflineGraph<NodeDataType, EdgeDataType>& srcGraph) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    globalNodeOffset.resize(numHosts);
    localNodeSize.resize(numHosts);
    std::vector<std::vector<uint64_t>> threadNodesToSend(
        galois::runtime::activeThreads);
    for (uint32_t i = 0; i < galois::runtime::activeThreads; i++) {
      threadNodesToSend[i].resize(numHosts, 0);
      I_WR();
    }
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      uint64_t beginNode;
      uint64_t endNode;
      std::tie(beginNode, endNode) = galois::block_range(
          (uint64_t)0, srcGraph.localNodes.size(), tid, nthreads);

      for (uint64_t i = beginNode; i < endNode; ++i) {
        int host =
            virtualToPhyMapping[srcGraph.localNodes[i].glbid % numVirtualHosts];
        threadNodesToSend[tid][host]++;
        I_WR();
        for (int k = 0; k < 2; k++)
          I_RR();
      }
    });
    for (uint32_t tid = 0; tid < galois::runtime::activeThreads; tid++) {
      for (uint32_t h = 0; h < numHosts; h++) {
        localNodeSize[h] += threadNodesToSend[tid][h];
        I_RR();
        I_WR();
      }
    }

    numNodes = 0;

    // send vertex size to other hosts
    for (uint32_t h = 0; h < numHosts; ++h) {
      if (h == hostID) {
        continue;
      }
      // serialize size_t
      galois::runtime::SendBuffer sendBuffer;
      galois::runtime::gSerialize(sendBuffer, localNodeSize);
      net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
      I_WM(localNodeSize.size());
    }

    for (uint32_t h = 0; h < numHosts - 1; h++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      std::vector<uint64_t> cnt;
      // deserialize local_node_size
      galois::runtime::gDeserialize(p->second, cnt);
      I_LC(p->first, cnt.size() * sizeof(uint64_t));
      for (uint32_t i = 0; i < numHosts; i++) {
        localNodeSize[i] += cnt[i];
        I_RR();
        I_WR();
      }
    }

    numNodes      = localNodeSize[hostID];
    numLocalNodes = numNodes;
    // compute prefix sum to get offset
    globalNodeOffset[0] = 0;
    for (size_t h = 1; h < numHosts; h++) {
      globalNodeOffset[h] = localNodeSize[h - 1] + globalNodeOffset[h - 1];
      for (int k = 0; k < 2; k++)
        I_RR();
      I_WR();
    }
    srcGraph.setSize(globalNodeOffset[numHosts - 1] +
                     localNodeSize[numHosts - 1]);
    for (int k = 0; k < 2; k++)
      I_RR();
    I_WR();

    increment_evilPhase();
  }

  /**
   * Exchanges vertex ids to form a global id to local id map before exchanging
   * edges so that using the map edges can be inserted into the edgelist
   */
  void
  gatherVerticesAndEdges(std::vector<std::vector<EdgeDataType>>& localEdges,
                         std::vector<NodeDataType>& localNodes) {
    auto& net              = galois::runtime::getSystemNetworkInterface();
    uint32_t activeThreads = galois::getActiveThreads();

    // prepare both nodedata and edgedata to send to all hosts
    std::vector<std::vector<std::vector<EdgeDataType>>> edgesToSend(
        numHosts, std::vector<std::vector<EdgeDataType>>());
    std::vector<std::vector<NodeDataType>> nodesToSend(
        numHosts, std::vector<NodeDataType>());

    // PerThread DS
    std::vector<std::vector<std::vector<std::vector<EdgeDataType>>>>
        threadEdgesToSend(
            activeThreads,
            std::vector<std::vector<std::vector<EdgeDataType>>>());
    std::vector<std::vector<std::vector<NodeDataType>>> threadNodesToSend(
        activeThreads, std::vector<std::vector<NodeDataType>>());
    for (uint32_t i = 0; i < activeThreads; i++) {
      threadEdgesToSend[i].resize(numHosts);
      threadNodesToSend[i].resize(numHosts);
    }

    // Prepare edgeList and Vertex ID list to send to other hosts
    uint64_t sz = localEdges.size();
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      uint64_t beginNode;
      uint64_t endNode;
      std::tie(beginNode, endNode) =
          galois::block_range((uint64_t)0, sz, tid, nthreads);

      for (uint64_t i = beginNode; i < endNode; ++i) {
        uint64_t src = localEdges[i][0].src;
        int host     = virtualToPhyMapping[src % numVirtualHosts];
        threadEdgesToSend[tid][host].push_back((localEdges[i]));
        for (int k = 0; k < 3; k++)
          I_RR();
        I_WM(2);
      }
    });

    // Prepare Nodedata to send to other hosts
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      size_t beginNode;
      size_t endNode;
      std::tie(beginNode, endNode) =
          galois::block_range((uint64_t)0, localNodes.size(), tid, nthreads);

      for (size_t i = beginNode; i < (endNode); ++i) {
        int host = virtualToPhyMapping[(localNodes[i].glbid) %
                                       (scaleFactor * numHosts)];
        threadNodesToSend[tid][host].push_back((localNodes[i]));
        for (int k = 0; k < 2; k++)
          I_RR();
        I_WR();
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
        for (int i = 0; i < 6; i++)
          I_RR();
        I_WM(3);
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
      net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
      I_WM(nodesToSend[h].size());
    }

    // Collect node data received from other hosts
    for (uint32_t i = 0; i < (numHosts - 1); i++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      std::vector<NodeDataType> NodeData;
      galois::runtime::gDeserialize(p->second, NodeData);
      I_LC(p->first, NodeData.size() * sizeof(NodeDataType));
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
        for (size_t j = beginNode; j < (endNode); ++j) {
          threadMap[tid][NodeData[j].glbid] =
              offset + (tid * (delta)) + j - beginNode;
          threadLIDMap[tid][offset + (tid * (delta)) + j - beginNode] =
              NodeData[j].glbid;
          I_WR();
          I_RR();
        }
      });
      for (uint32_t t = 0; t < activeThreads; t++) {
        GIDtoLID.insert(threadMap[t].begin(), threadMap[t].end());
        LIDtoGID.insert(threadLIDMap[t].begin(), threadLIDMap[t].end());
        I_WR();
        I_RR();
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
      for (size_t i = beginNode; i < (endNode); ++i) {
        threadMap[tid][nodesToSend[hostID][i].glbid] =
            offset + (tid * (delta)) + i - beginNode;
        threadLIDMap[tid][offset + (tid * (delta)) + i - beginNode] =
            nodesToSend[hostID][i].glbid;
        I_WR();
        I_RR();
      }
    });
    for (uint32_t t = 0; t < activeThreads; t++) {
      GIDtoLID.insert(threadMap[t].begin(), threadMap[t].end());
      LIDtoGID.insert(threadLIDMap[t].begin(), threadLIDMap[t].end());
      I_WR();
      I_RR();
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
      net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
      I_WM(edgesToSend[h].size());
    }

    // Appending edges in each host that belong to self
    localEdges.resize(GIDtoLID.size());

    // Receiving edges from other hosts and populating edgelist
    for (uint32_t h = 0; h < (numHosts - 1); h++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      uint32_t sendingHost = p->first;

      std::vector<std::vector<EdgeDataType>> edgeList;

      galois::runtime::gDeserialize(p->second, edgeList);
#ifdef GALOIS_INSTRUMENT
      for (auto l : edgeList)
        I_LC(sendingHost, l.size() * sizeof(EdgeDataType));
#endif

      galois::gInfo("[", hostID, "] recv from ", sendingHost,
                    " edgeList size: ", edgeList.size());

      galois::on_each([&](unsigned tid, unsigned nthreads) {
        size_t beginNode;
        size_t endNode;
        std::tie(beginNode, endNode) =
            galois::block_range((size_t)0, edgeList.size(), tid, nthreads);
        for (size_t j = beginNode; j < endNode; j++) {
          auto lid = GIDtoLID[edgeList[j][0].src];
          localEdges[lid].insert(std::end(localEdges[lid]),
                                 std::begin(edgeList[j]),
                                 std::end(edgeList[j]));
          for (int i = 0; i < 3; i++)
            I_RR();
          I_WR();
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
        localEdges[lid].insert(std::end(localEdges[lid]),
                               std::begin(edgesToSend[hostID][j]),
                               std::end(edgesToSend[hostID][j]));
        for (int i = 0; i < 4; i++)
          I_RR();
        I_WR();
      }
    });
    edgesToSend.clear();
    increment_evilPhase();
  }

  /**
   * Flatten the 2D vector localEdges into a CSR edge list
   * Will compute edge size and build CSR edge offset mapping
   */
  void flattenEdges(std::vector<std::vector<EdgeDataType>>& localEdges) {
    // build CSR edge offseto
    offsets.resize(numLocalNodes + 1, 0);
    for (size_t i = 0; i < numLocalNodes; i++) {
      uint64_t cnt;
      if (i >= localEdges.size())
        cnt = 0;
      else
        cnt = localEdges[i].size();
      offsets[i + 1] += cnt + offsets[i];
      I_RR();
      I_WR();
    }
    numLocalEdges = offsets[numLocalNodes];

    // build flatten edge list
    edges.resize(numLocalEdges);
    galois::do_all(
        galois::iterate((size_t)0, localEdges.size()),
        [this, &localEdges](size_t i) {
          I_WM(localEdges[i].size());
          std::move(localEdges[i].begin(), localEdges[i].end(),
                    edges.begin() + offsets[i]);
        },
        galois::steal());
  }

public:
  WMDBufferedGraph() : BufferedGraph<EdgeDataType>() {}
  std::unordered_map<uint64_t, uint32_t> GIDtoLID;
  std::unordered_map<uint32_t, uint64_t> LIDtoGID;

  // copy not allowed
  //! disabled copy constructor
  WMDBufferedGraph(const WMDBufferedGraph&) = delete;
  //! disabled copy constructor operator
  WMDBufferedGraph& operator=(const WMDBufferedGraph&) = delete;
  // move not allowed
  //! disabled move operator
  WMDBufferedGraph(WMDBufferedGraph&&) = delete;
  //! disabled move constructor operator
  WMDBufferedGraph& operator=(WMDBufferedGraph&&) = delete;

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
   * @param nodeStart First node to load
   * @param nodeEnd Last node to load, non-inclusive
   * @param numGlobalNodes Total number of nodes in the graph
   * @param numGlobalEdges Total number of edges in the graph
   */
  void loadPartialGraph(WMDOfflineGraph<NodeDataType, EdgeDataType>& srcGraph,
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
      I_RS();
      I_WR();
    }

    // build local buffered graph
    exchangeLocalNodeSize(srcGraph);
    galois::gDebug("[", hostID, "] gatherVerticesAndEdges!");
    gatherVerticesAndEdges(srcGraph.localEdges, srcGraph.localNodes);
    galois::gDebug("[", hostID, "] ", "flattenEdges!");
    flattenEdges(srcGraph.localEdges);

    // clean unused data
    srcGraph.localEdgesIdxToGlobalNodeID.clear();
    srcGraph.localEdgesIdxToGlobalNodeID.shrink_to_fit();
    srcGraph.localEdges.clear();
    srcGraph.localEdges.shrink_to_fit();

    graphLoaded = true;

    galois::gDebug("[", hostID, "] ", "exchangeNodeRange!");
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
  template <typename GraphTy>
  void gatherNodes(WMDOfflineGraph<NodeDataType, EdgeDataType>& srcGraph,
                   GraphTy& dstGraph,
                   std::vector<std::vector<uint64_t>>& proxiesOnHosts,
                   uint64_t totalLocalNodes,
                   std::unordered_map<uint64_t, uint32_t> globalToLocalMap) {
#ifdef NDEBUG
    (void)totalLocalNodes;
#endif
    auto& net        = galois::runtime::getSystemNetworkInterface();
    auto& localNodes = srcGraph.localNodes;

    // prepare data to send for all hosts
    // each host will receive its nodes and corresponding node global ID
    // list
    galois::gDebug("[", hostID, "] ", "prepare node data!");
    std::vector<std::vector<NodeDataType>> nodesToSend(
        numHosts, std::vector<NodeDataType>());
    std::vector<std::vector<std::vector<NodeDataType>>> threadNodesToSend(
        galois::runtime::activeThreads,
        std::vector<std::vector<NodeDataType>>());
    uint64_t globalIDOffset = 0;

    for (uint32_t i = 0; i < galois::runtime::activeThreads; i++) {
      threadNodesToSend[i].resize(numHosts);
      I_WR();
    }
    // Phase 1
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      size_t beginNode;
      size_t endNode;
      std::tie(beginNode, endNode) = galois::block_range(
          (size_t)0, srcGraph.localNodes.size(), tid, nthreads);

      for (size_t i = beginNode; i < endNode; ++i) {
        int host =
            virtualToPhyMapping[srcGraph.localNodes[i].id % numVirtualHosts];
        threadNodesToSend[tid][host].push_back((srcGraph.localNodes[i]));
        for (int k = 0; k < 2; k++)
          I_RR();
        I_WR();
      }
    });

    for (uint32_t tid = 0; tid < galois::runtime::activeThreads; tid++) {
      for (uint32_t h = 0; h < numHosts; h++) {
        nodesToSend[h].insert(nodesToSend[h].end(),
                              threadNodesToSend[tid][h].begin(),
                              threadNodesToSend[tid][h].end());
        for (int k = 0; k < 2; k++)
          I_RR();
        I_WR();
      }
    }
    srcGraph.localNodes.clear();

    // Send Nodedata only
    for (uint32_t h = 0; h < numHosts; h++) {
      if (h == hostID)
        continue;
      galois::runtime::SendBuffer sendBuffer;
      galois::runtime::gSerialize(sendBuffer, nodesToSend[h]);
      net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
      I_WM(nodesToSend[h].size());
    }
#ifndef NDEBUG
    std::atomic<uint64_t> addedData{0};
#endif
    for (uint32_t i = 0; i < (numHosts - 1); i++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      std::vector<NodeDataType> NodeData;
      I_LC(p->first, p->second.size());
      galois::runtime::gDeserialize(p->second, NodeData);
      galois::do_all(galois::iterate((size_t)0, NodeData.size()),
                     [this, NodeData, &dstGraph, &globalToLocalMap
#ifndef NDEBUG
                      ,
                      &addedData
#endif
      ](size_t j) {
                       dstGraph.getData(GIDtoLID[NodeData[j].id]) = NodeData[j];
                     });
      NodeData.clear();
    }
    galois::do_all(galois::iterate((size_t)0, nodesToSend[hostID].size()),
                   [this, nodesToSend, &dstGraph, &globalToLocalMap
#ifndef NDEBUG
                    ,
                    &addedData
#endif
    ](size_t i) {
                     dstGraph.getData(GIDtoLID[nodesToSend[hostID][i].id]) =
                         nodesToSend[hostID][i];
#ifndef NDEBUG
                     addedData++;
#endif
                   });
    nodesToSend.clear();
    nodesToSend.resize(numHosts);
    increment_evilPhase();

    // Phase 2
    //     uint64_t numNodes = srcGraph.localNodeSize[hostID];
    galois::do_all(
        galois::iterate((uint64_t)0, (uint64_t)numHosts),
        [this, &nodesToSend, &localNodes, &proxiesOnHosts, globalIDOffset,
         &dstGraph, &globalToLocalMap](uint64_t i) {
          if (i != hostID) {
            I_RR();
            for (uint64_t j = 0; j < proxiesOnHosts[i].size(); j++) {
              auto& r =
                  dstGraph.getData(globalToLocalMap[proxiesOnHosts[i][j]]);
              nodesToSend[i].push_back(r);
            };
          }
        },
        galois::steal());

    // send nodes to other hosts
    galois::gDebug("[", hostID, "] ", "send nodes!");

    for (uint32_t h = 0; h < numHosts; h++) {
      if (h == hostID)
        continue;
      assert(nodesToSend[h].size() == proxiesOnHosts[h].size());
      galois::runtime::SendBuffer sendBuffer;
      galois::runtime::gSerialize(sendBuffer, nodesToSend[h]);
      galois::runtime::gSerialize(sendBuffer, proxiesOnHosts[h]);
      galois::gDebug("[", hostID, "] ", "send to ", h,
                     " nodesToSend size: ", nodesToSend[h].size());
      I_WM(nodesToSend[h].size() + proxiesOnHosts[h].size());
      net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
    }

    nodesToSend.clear();
    // recive nodes from other hosts
    for (uint32_t i = 0; i < (numHosts - 1); i++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      uint32_t sendingHost = p->first;

      std::vector<NodeDataType> nodeRecv;
      std::vector<uint64_t> IDofNodeRecv;
      I_LC(sendingHost, p->second.size());
      galois::runtime::gDeserialize(p->second, nodeRecv);
      galois::runtime::gDeserialize(p->second, IDofNodeRecv);

      assert(nodeRecv.size() == IDofNodeRecv.size());
      galois::gDebug("[", hostID, "] recv from ", sendingHost,
                     " nodeRecv size: ", nodeRecv.size());

      galois::do_all(
          galois::iterate((size_t)0, IDofNodeRecv.size()),
          [this, &nodeRecv, &IDofNodeRecv, &dstGraph,
           &globalToLocalMap](size_t j) {
            dstGraph.getData(globalToLocalMap[IDofNodeRecv[j]]) = nodeRecv[j];
            for (int k = 0; k < 2; k++)
              I_RR();
            I_WR();
          },
          galois::steal());
      nodeRecv.clear();
      IDofNodeRecv.clear();
    }
#ifndef NDEBUG
    //    assert(addedData == totalLocalNodes);
#endif

    increment_evilPhase();

    // clean unused memory
    srcGraph.localNodes.clear();
    srcGraph.localNodes.shrink_to_fit();
    srcGraph.nodeOffset.clear();
    srcGraph.nodeOffset.shrink_to_fit();
  }

  // NOTE: for below methods, it return local edge id instead of global id

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
    return offsets[globalNodeID - globalNodeOffset[hostID] + 1] -
           offsets[globalNodeID - globalNodeOffset[hostID]];
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
    auto end = offsets[globalNodeID - globalNodeOffset[hostID] + 1];
    for (auto itr = offsets[globalNodeID - globalNodeOffset[hostID]];
         itr != end; ++itr) {
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
