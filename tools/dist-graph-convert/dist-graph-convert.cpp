/** Distributed graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 *
 * Distributed graph converter tool based on shared-memory graph convert.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include <fstream>
#include <utility>
#include <mutex>
#include <mpi.h>

#include "galois/Endian.h"
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Network.h"
#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;

// TODO: move these enums to a common location for all graph convert tools
enum ConvertMode {
  edgelist2gr,
  edgelistb2gr,
};

enum EdgeType {
  float32_,
  float64_,
  int32_,
  int64_,
  uint32_,
  uint64_,
  void_
};

static cll::opt<std::string> inputFilename(cll::Positional, 
    cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional,
    cll::desc("<output file>"), cll::Required);
static cll::opt<EdgeType> edgeType("edgeType", 
    cll::desc("Input/Output edge type:"),
    cll::values(
      clEnumValN(EdgeType::float32_, "float32", 
                 "32 bit floating point edge values"),
      clEnumValN(EdgeType::float64_, "float64", 
                 "64 bit floating point edge values"),
      clEnumValN(EdgeType::int32_, "int32", 
                 "32 bit int edge values"),
      clEnumValN(EdgeType::int64_, "int64", 
                 "64 bit int edge values"),
      clEnumValN(EdgeType::uint32_, "uint32", 
                 "32 bit unsigned int edge values"),
      clEnumValN(EdgeType::uint64_, "uint64", 
                 "64 bit unsigned int edge values"),
      clEnumValN(EdgeType::void_, "void", 
                 "no edge values"),
      clEnumValEnd), 
    cll::init(EdgeType::void_));
static cll::opt<ConvertMode> convertMode(cll::desc("Conversion mode:"),
    cll::values(
      clEnumVal(edgelistb2gr, "Convert edge list binary to binary gr"),
      clEnumVal(edgelist2gr, "Convert edge list to binary gr"),
      clEnumValEnd
    ), cll::Required);
static cll::opt<unsigned> totalNumNodes("numNodes", 
                                        cll::desc("Nodes in input graph"),
                                        cll::init(0));
static cll::opt<unsigned> threadsToUse("t", cll::desc("Threads to use"),
                                       cll::init(1));


// Base structures to inherit from: name specifies what the converter can do
struct Conversion { };
struct HasOnlyVoidSpecialization { };
struct HasNoVoidSpecialization { };

/**
 * Convert 1: figure out edge type, then call convert with edge type as
 * an additional template argument.
 */
template<typename C>
void convert() {
  C c;

  switch (edgeType) {
    case EdgeType::float32_: convert<float>(c, c); break;
    case EdgeType::float64_: convert<double>(c, c); break;
    case EdgeType::int32_: convert<int32_t>(c, c); break;
    case EdgeType::int64_: convert<int64_t>(c, c); break;
    case EdgeType::uint32_: convert<uint32_t>(c, c); break;
    case EdgeType::uint64_: convert<uint64_t>(c, c); break;
    case EdgeType::void_: convert<void>(c, c); break;
    default: abort();
  };
}

/**
 * Convert 2 called from convert above: calls convert from the appropriate
 * structure
 */
template<typename EdgeTy, typename C>
void convert(C& c, Conversion) {
  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    galois::gPrint("Input: ", inputFilename, "; Output: ", outputFilename, 
                   "\n");
  }

  c.template convert<EdgeTy>(inputFilename, outputFilename);

  if (net.ID == 0) {
    galois::gPrint("Done with convert\n");
  }
}

/**
 * Wrapper for MPI calls that return an error code. Make sure it is success
 * else die.
 *
 * @param errcode error code returned by an mpi call
 */
void MPICheck(int errcode) {
  if (errcode != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, errcode);
  }
}

/**
 * Wrapper for MPI file open, read only mode
 */
void openFileMPI(const std::string& filename, MPI_File* returnFile) {
  MPICheck(MPI_File_open(MPI_COMM_WORLD, filename.c_str(),
                         MPI_MODE_RDONLY, MPI_INFO_NULL, 
                         returnFile));
}

std::vector<std::pair<uint64_t, uint64_t>> getHostToNodeMapping(
    const uint64_t numHosts
) {
  GALOIS_ASSERT((totalNumNodes != 0), "host2node mapping needs numNodes");

  std::vector<std::pair<uint64_t, uint64_t>> hostToNodes;

  for (unsigned i = 0; i < numHosts; i++) {
    hostToNodes.emplace_back(
      galois::block_range((uint64_t)0, (uint64_t)totalNumNodes, i, numHosts)
    );
  }

  return hostToNodes;
}

/**
 * Get the assigned host of some node given its global id.
 *
 * @param gID global ID of a node
 * @param hostToNodes Vector containing information about which host has which
 * nodes
 * @returns Host that requested node resides on or -1 if it couldn't be found
 */
uint32_t findHostID(const uint64_t gID, 
            const std::vector<std::pair<uint64_t, uint64_t>> hostToNodes) {
  for (uint64_t host = 0; host < hostToNodes.size(); host++) {
    if (gID >= hostToNodes[host].first && gID < hostToNodes[host].second) {
      return host;
    }
  }
  return -1;
}

/**
 * Returns the file size of an ifstream.
 *
 * @param openFile an open ifstream
 * @returns file size in bytes of the ifstream
 */
uint64_t getFileSize(std::ifstream& openFile) {
  openFile.seekg(0, std::ios_base::end);
  return openFile.tellg();
}

// TODO incomplete implementation
struct EdgelistB2Gr : public Conversion {
  /**
   * Gets total number of edges in a binary edge file. Assumes
   * no weights + edges size uint32_t
   *
   * @param edgeBinaryFile file to get num edges; will alter the seek position
   * @returns total number of edges in the binary edge list
   */
  uint64_t getTotalNumEdges(MPI_File& edgeBinaryFile) {
    MPI_Offset fileSize;
    MPICheck(MPI_File_get_size(edgeBinaryFile, &fileSize));
    uint64_t edgeSize = sizeof(uint32_t) * 2;
    GALOIS_ASSERT(((fileSize % edgeSize) == 0), "file size ", fileSize, 
                  "should be divisible by assumed edge size ", edgeSize);
    return fileSize / edgeSize;
  }

  /**
   * Reads a chunk of edge data from the binary edge list into an array.
   *
   * @param edgeBinaryFile file opened to the binary edge list
   * @param localNumEdges total number of edges to read from the file
   * @param localEdgeBegin the first edge to read from
   * @param localEdgeArray the array to read into (this is output)
   */
  void readDataFromEdgelist(MPI_File& edgeBinaryFile, uint64_t localNumEdges, 
                            uint64_t localEdgeBegin, uint32_t* localEdgeArray) {
    if (localNumEdges == 0) {
      return;
    }

    const uint64_t localStartPosition = localEdgeBegin * 2 * sizeof(uint32_t);
    uint64_t edgeHalvesRead = 0;
    uint64_t edgeHalvesToRead = 2 * localNumEdges;
    MPI_Status readStatus;
    while (edgeHalvesToRead > 0) {
      MPI_File_read_at(edgeBinaryFile, 
                       localStartPosition + (edgeHalvesRead * sizeof(uint32_t)), 
                       (char*)(localEdgeArray + edgeHalvesRead), 
                       edgeHalvesToRead, MPI_UINT32_T, &readStatus);
                
      int itemsRead; 
      MPI_Get_count(&readStatus, MPI_UINT32_T, &itemsRead);

      edgeHalvesRead += itemsRead;
      edgeHalvesToRead -= itemsRead;
    }
  }


  // TODO doesn't handle 0 node/edge case
  template<typename EdgeTy>
  void convert(const std::string& inputFile, const std::string& outputFile) {
    GALOIS_ASSERT((totalNumNodes != 0), "edgelist binary2gr needs num nodes");
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;
    uint64_t totalNumHosts = net.Num;

    MPI_File edgeBinaryFile;
    MPICheck(MPI_File_open(MPI_COMM_WORLD, inputFile.c_str(),
                           MPI_MODE_RDONLY, MPI_INFO_NULL, 
                           &edgeBinaryFile));
    uint64_t totalNumberOfEdges = getTotalNumEdges(edgeBinaryFile);
    if (hostID == 0) {
      printf("Total number of edges is %lu\n", totalNumberOfEdges);
    }

    uint64_t localEdgeBegin = 0;
    uint64_t localEdgeEnd = 0;
    std::tie(localEdgeBegin, localEdgeEnd) = galois::block_range((uint64_t)0, 
                                             totalNumberOfEdges, hostID, 
                                             totalNumHosts);
    printf("[%lu] Edges %lu to %lu\n", hostID, localEdgeBegin, 
                                           localEdgeEnd);
    uint64_t localNumEdges = localEdgeEnd - localEdgeBegin;
    uint32_t* localEdgeArray = (uint32_t*)malloc(2 * sizeof(uint32_t) *
                                                localNumEdges);
    readDataFromEdgelist(edgeBinaryFile, localNumEdges, localEdgeBegin, 
                         localEdgeArray);
    //printf("local edge %lu %lu\n", localEdgeArray[0], localEdgeArray[1]);
    // done with input file; close it
    MPICheck(MPI_File_close(&edgeBinaryFile));

    // split up nodes among hosts
    std::vector<std::pair<uint64_t, uint64_t>> hostToNodes = 
        getHostToNodeMapping(totalNumHosts);

    uint64_t localNodeBegin = hostToNodes[hostID].first;
    uint64_t localNodeEnd = hostToNodes[hostID].second;
    uint64_t localNumNodes = localNodeEnd - localNodeBegin;

    printf("[%lu] Nodes %lu to %lu\n", hostID, localNodeBegin, localNodeEnd);

    using NodeArraysType = std::vector<std::vector<uint32_t>>;
    
    // each assigned node gets a vector; slow once emplace_backs start 
    // happening... TODO change this?
    NodeArraysType nodeToDestinations(localNumNodes);

    // pass + send all edges off

    using DestinationVectorType = std::vector<std::vector<uint32_t>>;
    galois::substrate::PerThreadStorage<DestinationVectorType> 
        destBuffers(totalNumHosts);
    using SendBufferVectorTy = std::vector<galois::runtime::SendBuffer>;
    galois::substrate::PerThreadStorage<SendBufferVectorTy> 
        sendBuffers(totalNumHosts);

    // pass 1: determine to which host each edge will go
    galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&] (uint64_t edgeIndex) {
        uint32_t src = localEdgeArray[edgeIndex * 2];
        uint32_t dst = localEdgeArray[(edgeIndex * 2) + 1];

        uint32_t edgeOwner = findHostID(src, hostToNodes);

        //DestinationVectorType& localDestBuffer = *(destBuffers.getLocal())
        //localDestBuffer[edgeOwner]

        //auto& localSendBuffer = *(sendBuffers.getLocal());

        //sendBuffers.
      },
      galois::loopname("EdgeSending"),
      galois::no_stats(),
      galois::steal<false>(),
      galois::timeit()
    );

    // comm 1: tell hosts how many edges they should reserve space for
    //sendEdgeCountsPerHost();

    // pass 2/comm 2: actually send out edges to hosts on this second go around
    //sendAndReceiveEdges();

    // hosts have all their edges for their assigned nodes: write into buffers
    // (if not already done)
    // TODO prefix sum of edges, allocate node index array, allocate edge array
    // if not already done after comm 1

    // comm 3: send total number of edges I (this host) has to all other hosts
    // so hosts can determine offsets to write into
    //determineEdgeDistribution();

    // write out into a file in parallel with other processes (MPI write): write
    // into correct location

    free(localEdgeArray);

    // TODO free other used buffers

    galois::runtime::getHostBarrier().wait();
  }
};

struct Edgelist2Gr : public Conversion {
  /**
   * TODO
   */
  std::pair<uint64_t, uint64_t> determineByteRange(std::ifstream& edgeListFile,
                                                   uint64_t fileSize,
                                                   uint64_t hostID,
                                                   uint64_t totalNumHosts) {
    uint64_t initialStart;
    uint64_t initialEnd;
    std::tie(initialStart, initialEnd) = galois::block_range((uint64_t)0, 
                                                             (uint64_t)fileSize,
                                                             hostID, 
                                                             totalNumHosts);
    bool startGood = false;
    if (initialStart != 0) {
      // good starting point if the prev char was a new line (i.e. this start
      // location is the beginning of a line)
      // TODO factor this out
      edgeListFile.seekg(initialStart - 1);
      char testChar = edgeListFile.get();
      if (testChar == '\n') {
        startGood = true;
      }
    } else {
      // start is 0; perfect starting point, need no adjustment
      startGood = true;
    }

    bool endGood = false;
    if (initialEnd != fileSize && initialEnd != 0) {
      // good end point if the prev char was a new line (i.e. this end
      // location is the beginning of a line; recall non-inclusive)
      // TODO factor this out
      edgeListFile.seekg(initialEnd - 1);
      char testChar = edgeListFile.get();
      if (testChar == '\n') {
        endGood = true;
      }
    } else {
      endGood = true;
    }

    uint64_t finalStart = initialStart;
    if (!startGood) {
      // find next new line
      // TODO factor this out
      edgeListFile.seekg(initialStart);
      std::string dummy;
      std::getline(edgeListFile, dummy);
      finalStart = edgeListFile.tellg();
    }

    uint64_t finalEnd = initialEnd;
    if (!endGood) {
      // find next new line
      // TODO factor out
      edgeListFile.seekg(initialEnd);
      std::string dummy;
      std::getline(edgeListFile, dummy);
      finalEnd = edgeListFile.tellg();
    }

    return std::pair<uint64_t, uint64_t>(finalStart, finalEnd);
  }

  /**
   * Determine/send to each host how many edges they should expect to receive
   * from the caller.
   *
   * TODO params
   */
  void sendEdgeCounts(
      const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
      uint64_t localNumEdges, const std::vector<uint32_t>& localEdges)
  {
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;
    uint64_t totalNumHosts = net.Num;

    std::vector<galois::GAccumulator<uint64_t>> numEdgesPerHost(totalNumHosts);

    // determine to which host each edge will go
    galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&] (uint64_t edgeIndex) {
        uint32_t src = localEdges[edgeIndex * 2];
        uint32_t edgeOwner = findHostID(src, hostToNodes);
        numEdgesPerHost[edgeOwner] += 1;
      },
      galois::loopname("EdgeInspection"),
      galois::no_stats(),
      galois::steal<false>(),
      galois::timeit()
    );

    // tell hosts how many edges they should expect
    std::vector<uint64_t> edgeVectorToSend;
    for (unsigned h = 0; h < totalNumHosts; h++) {
      if (h == hostID) continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, numEdgesPerHost[h].reduce());
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
  }

  /**
   * Receive the messages from other hosts that tell this host how many edges
   * it should expect to receive.
   *
   * TODO params
   */
  uint64_t receiveEdgeCounts(uint64_t localNumNodes) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;
    uint64_t totalNumHosts = net.Num;

    uint64_t edgesToReceive = 0;

    // receive
    for (unsigned h = 0; h < totalNumHosts; h++) {
      if (h == hostID) continue;
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) rBuffer;

      uint64_t recvCount;

      do {
        rBuffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!rBuffer);
      galois::runtime::gDeserialize(rBuffer->second, recvCount);

      edgesToReceive += recvCount;
    }

    galois::runtime::evilPhase++;

    return edgesToReceive;
  }

  /**
   * TODO
   */
  void sendAssignedEdges( 
      const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
      uint64_t localNumEdges, const std::vector<uint32_t>& localEdges,
      std::vector<std::vector<uint32_t>>& localSrcToDest,
      std::vector<std::mutex>& nodeLocks)
  {
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;
    uint64_t totalNumHosts = net.Num;

    using DestVectorTy = std::vector<std::vector<uint32_t>>;
    galois::substrate::PerThreadStorage<DestVectorTy> 
        dstVectors(totalNumHosts);
    using SendBufferVectorTy = std::vector<galois::runtime::SendBuffer>;
    galois::substrate::PerThreadStorage<SendBufferVectorTy> 
        sendBuffers(totalNumHosts);
    galois::substrate::PerThreadStorage<std::vector<uint64_t>> 
        lastSourceSentStorage(totalNumHosts);

    // initialize last source sent
    galois::on_each(
      [&] (unsigned tid, unsigned nthreads) {
        for (unsigned h = 0; h < totalNumHosts; h++) {
          (*(lastSourceSentStorage.getLocal()))[h] = 0;
        }
      },
      galois::no_stats()
    );

    // pass 1: determine to which host each edge will go
    galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&] (uint64_t edgeIndex) {
        uint32_t src = localEdges[edgeIndex * 2];
        uint32_t edgeOwner = findHostID(src, hostToNodes);
        uint32_t dst = localEdges[(edgeIndex * 2) + 1];
        uint32_t localID = src - hostToNodes[edgeOwner].first;

        if (edgeOwner != hostID) {
          // send off to correct host
          auto& hostSendBuffer = (*(sendBuffers.getLocal()))[edgeOwner];
          auto& dstVector = (*(dstVectors.getLocal()))[edgeOwner];
          auto& lastSourceSent = 
              (*(lastSourceSentStorage.getLocal()))[edgeOwner];

          if (lastSourceSent == localID) {
            dstVector.emplace_back(dst);
          } else {
            // serialize vector if anything exists in it + send buffer if
            // reached some limit
            if (dstVector.size() > 0) {
              uint64_t globalSourceID = lastSourceSent + 
                                        hostToNodes[edgeOwner].first;
              galois::runtime::gSerialize(hostSendBuffer, globalSourceID, 
                                          dstVector);
              dstVector.clear();
              if (hostSendBuffer.size() > 1400) {
                net.sendTagged(edgeOwner, galois::runtime::evilPhase, 
                               hostSendBuffer);
                hostSendBuffer.getVec().clear();
              }
            }

            dstVector.emplace_back(dst);
            lastSourceSent = localID;
          }
        } else {
          // save to edge dest array
          nodeLocks[localID].lock();
          localSrcToDest[localID].emplace_back(dst);
          nodeLocks[localID].unlock();
        }
      },
      galois::loopname("Pass2"),
      galois::no_stats(),
      galois::steal<false>(),
      galois::timeit()
    );

    // cleanup: each thread serialize + send out remaining stuff
    galois::on_each(
      [&] (unsigned tid, unsigned nthreads) {
        for (unsigned h = 0; h < totalNumHosts; h++) {
          if (h == hostID) continue;
          auto& hostSendBuffer = (*(sendBuffers.getLocal()))[h];
          auto& dstVector = (*(dstVectors.getLocal()))[h];
          uint64_t lastSourceSent = (*(lastSourceSentStorage.getLocal()))[h];

          if (dstVector.size() > 0) {
            uint64_t globalSourceID = lastSourceSent + 
                                      hostToNodes[h].first;
            galois::runtime::gSerialize(hostSendBuffer, globalSourceID,
                                        dstVector);
            dstVector.clear();
          }

          if (hostSendBuffer.size() > 0) {
            net.sendTagged(h, galois::runtime::evilPhase, hostSendBuffer);
            hostSendBuffer.getVec().clear();
          }
        }
      },
      galois::loopname("Pass2Cleanup"),
      galois::timeit(),
      galois::no_stats()
    );
  }

  void receiveAssignedEdges(std::atomic<uint64_t>& edgesToReceive,
      const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
      std::vector<std::vector<uint32_t>>& localSrcToDest,
      std::vector<std::mutex>& nodeLocks)
  {
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;

    // receive edges
    galois::on_each(
      [&] (unsigned tid, unsigned nthreads) {
        std::vector<uint32_t> recvVector;
        while (edgesToReceive) {
          decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) 
              rBuffer;
          rBuffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
          
          if (rBuffer) {
            auto& receiveBuffer = rBuffer->second;
            while (receiveBuffer.r_size() > 0) {
              uint64_t src;
              galois::runtime::gDeserialize(receiveBuffer, src, recvVector);
              edgesToReceive -= recvVector.size();
              GALOIS_ASSERT(findHostID(src, hostToNodes) == hostID);
              uint32_t localID = src - hostToNodes[hostID].first;

              nodeLocks[localID].lock();
              for (unsigned i = 0; i < recvVector.size(); i++) {
                localSrcToDest[localID].emplace_back(recvVector[i]);
              }
              nodeLocks[localID].unlock();
            }
          }
        }
      },
      galois::loopname("EdgeReceiving"),
      galois::timeit(),
      galois::no_stats()
    );
    galois::runtime::evilPhase++; 
  }

  /**
   * Send/receive other hosts number of assigned edges.
   *
   * TODO
   */
  std::vector<uint64_t> getEdgesPerHost(uint64_t localAssignedEdges) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;
    uint64_t totalNumHosts = net.Num;

    std::vector<uint64_t> edgesPerHost(totalNumHosts);

    for (unsigned h = 0; h < totalNumHosts; h++) {
      if (h == hostID) continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, localAssignedEdges);
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }

    // receive
    for (unsigned h = 0; h < totalNumHosts; h++) {
      if (h == hostID) {
        edgesPerHost[h] = localAssignedEdges;
        continue;
      }

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) rBuffer;
      uint64_t otherAssignedEdges;
      do {
        rBuffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!rBuffer);
      galois::runtime::gDeserialize(rBuffer->second, otherAssignedEdges);

      edgesPerHost[rBuffer->first] = otherAssignedEdges;
    }
    galois::runtime::evilPhase++; 

    return edgesPerHost;
  }
   

  // WARNING
  // WILL NOT WORK IF THE EDGE LIST HAS WEIGHTS
  template<typename EdgeTy>
  void convert(const std::string& inputFile, const std::string& outputFile) {
    GALOIS_ASSERT((totalNumNodes != 0), "edgelist2gr needs num nodes");
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint64_t hostID = net.ID;
    uint64_t totalNumHosts = net.Num;

    std::ifstream edgeListFile(inputFile.c_str());
    uint64_t fileSize = getFileSize(edgeListFile);
    if (hostID == 0) {
      printf("File size is %lu\n", fileSize);
    }

    uint64_t localStartByte;
    uint64_t localEndByte;
    std::tie(localStartByte, localEndByte) = determineByteRange(edgeListFile,
                                                                fileSize,
                                                                hostID,
                                                                totalNumHosts);

    printf("[%lu] Byte start %lu byte end %lu\n", hostID, localStartByte,
                                                  localEndByte);

    // load edges into a vector
    uint64_t localNumEdges = 0;
    edgeListFile.seekg(localStartByte);
    std::vector<uint32_t> localEdges; // v1 support only
    while ((uint64_t)(edgeListFile.tellg() + 1) != localEndByte) {
      uint32_t src;
      uint32_t dst;
      edgeListFile >> src >> dst;
      localEdges.emplace_back(src);
      localEdges.emplace_back(dst);
      localNumEdges++;
    }
    edgeListFile.close();
    GALOIS_ASSERT(localNumEdges == (localEdges.size() / 2));
    printf("[%lu] Local num edges %lu\n", hostID, localNumEdges);

    // split up nodes among hosts
    std::vector<std::pair<uint64_t, uint64_t>> hostToNodes = 
        getHostToNodeMapping(totalNumHosts);

    uint64_t localNodeBegin = hostToNodes[hostID].first;
    uint64_t localNodeEnd = hostToNodes[hostID].second;
    uint64_t localNumNodes = localNodeEnd - localNodeBegin;

    printf("[%lu] Nodes %lu to %lu\n", hostID, localNodeBegin, localNodeEnd);

    sendEdgeCounts(hostToNodes, localNumEdges, localEdges);
    std::atomic<uint64_t> edgesToReceive;
    edgesToReceive.store(receiveEdgeCounts(localNumNodes));

    printf("[%lu] Need to receive %lu edges\n", hostID, edgesToReceive.load());

    // FIXME ONLY V1 SUPPORT
    std::vector<std::vector<uint32_t>> localSrcToDest(localNumNodes);
    std::vector<std::mutex> nodeLocks(localNumNodes);

    sendAssignedEdges(hostToNodes, localNumEdges, localEdges, localSrcToDest,
                      nodeLocks);
    receiveAssignedEdges(edgesToReceive, hostToNodes, localSrcToDest, 
                         nodeLocks);

    // TODO can refactor to function
    uint64_t totalAssignedEdges = 0;
    for (unsigned i = 0; i < localNumNodes; i++) {
      totalAssignedEdges += localSrcToDest[i].size();
    }

    galois::DGAccumulator<uint64_t> ecAccumulator;
    ecAccumulator.reset();
    ecAccumulator += totalAssignedEdges;
    uint64_t totalEdgeCount = ecAccumulator.reduce();
    if (hostID == 0) {
      printf("Total number of edges is %lu\n", totalEdgeCount);
    }

    // prepare edge prefix sum for file writing
    std::vector<uint64_t> edgePrefixSum(localNumNodes);
    edgePrefixSum[0] = localSrcToDest[0].size();
    for (unsigned i = 1; i < localNumNodes; i++) {
      edgePrefixSum[i] = (edgePrefixSum[i - 1] + localSrcToDest[i].size());
    }

    //for (unsigned i = 0; i < localNumNodes; i++) {
    //  //printf("%lu\n", edgePrefixSum.data()[i]);
    //  //printf("%lu\n", edgePrefixSum[i]);
    //}


    // calculate global edge offset using edge counts from other hosts
    std::vector<uint64_t> edgesPerHost = getEdgesPerHost(totalAssignedEdges);
    uint64_t globalEdgeOffset = 0;
    for (unsigned h = 0; h < hostID; h++) {
      globalEdgeOffset += edgesPerHost[h];
    }
    printf("[%lu] Edge offset %lu\n", hostID, globalEdgeOffset);

    // update node index pointers using edge offset
    for (unsigned i = 0; i < localNumNodes; i++) {
      edgePrefixSum[i] = edgePrefixSum[i] + globalEdgeOffset;
    }

    // begin file writing stuff
    
    MPI_File newGR;
    MPICheck(MPI_File_open(MPI_COMM_WORLD, outputFile.c_str(), 
             MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &newGR));

    // write header info if host id is 0
    if (hostID == 0) {
      uint64_t version = 1;
      uint64_t sizeOfEdge = 0;
      uint64_t nn = totalNumNodes;
      uint64_t ne = totalEdgeCount;

      MPICheck(MPI_File_write_at(newGR, 0, &version, 1, MPI_UINT64_T, 
               MPI_STATUS_IGNORE));
      MPICheck(MPI_File_write_at(newGR, sizeof(uint64_t), &sizeOfEdge, 1, 
                                 MPI_UINT64_T, MPI_STATUS_IGNORE));
      MPICheck(MPI_File_write_at(newGR, sizeof(uint64_t) * 2, &nn, 1, 
                                 MPI_UINT64_T, MPI_STATUS_IGNORE));
      MPICheck(MPI_File_write_at(newGR, sizeof(uint64_t) * 3, &ne, 1, 
                                 MPI_UINT64_T, MPI_STATUS_IGNORE));
    }

    uint64_t headerSize = sizeof(uint64_t) * 4;

    // write node index data
    uint64_t nodeIndexStart = headerSize + (localNodeBegin * sizeof(uint64_t));
    // TODO make sure it writes all nodes (mpi might not; see status)....
    MPICheck(MPI_File_write_at(newGR, nodeIndexStart, edgePrefixSum.data(),
                               localNumNodes, MPI_UINT64_T, MPI_STATUS_IGNORE));
    
    uint64_t edgeDestOffset = headerSize + (totalNumNodes * sizeof(uint64_t)) +
                              globalEdgeOffset * sizeof(uint32_t);

    // write edge dests
    for (unsigned i = 0; i < localNumNodes; i++) {
      std::vector<uint32_t> currentDests = localSrcToDest[i];
      uint64_t numToWrite = currentDests.size();
      MPICheck(MPI_File_write_at(newGR, edgeDestOffset, currentDests.data(),
                                 numToWrite, MPI_UINT32_T, MPI_STATUS_IGNORE));

      edgeDestOffset += sizeof(uint32_t) * numToWrite;
    }

    MPI_File_close(&newGR);

    galois::runtime::getHostBarrier().wait();
  }
};

int main(int argc, char** argv) {
  galois::DistMemSys G;
  llvm::cl::ParseCommandLineOptions(argc, argv);
  galois::setActiveThreads(threadsToUse);

  switch (convertMode) {
    //case edgelistb2gr: 
    //  convert<EdgelistB2Gr>(); break;
    case edgelist2gr: 
      convert<Edgelist2Gr>(); break;
    default: abort();
  }
  return 0;
}
