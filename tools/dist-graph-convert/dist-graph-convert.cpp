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

#include <utility>
#include <mutex>

#include "galois/DistGalois.h"
#include "galois/DistAccumulator.h"
#include "llvm/Support/CommandLine.h"

#include "dist-graph-convert-helpers.h"

namespace cll = llvm::cl;

// TODO: move these enums to a common location for all graph convert tools
enum ConvertMode {
  edgelist2gr,
  edgelistb2gr // TODO
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



////////////////////////////////////////////////////////////////////////////////
// END HELPER IMPLEMENTATIONS
////////////////////////////////////////////////////////////////////////////////

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

struct Edgelist2Gr : public Conversion {
   

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
                                                                fileSize);

    printf("[%lu] Byte start %lu byte end %lu, num bytes %lu\n", hostID, 
                   localStartByte, localEndByte, localEndByte - localStartByte);

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
        getHostToNodeMapping(totalNumHosts, totalNumNodes);

    uint64_t localNodeBegin = hostToNodes[hostID].first;
    uint64_t localNodeEnd = hostToNodes[hostID].second;
    uint64_t localNumNodes = localNodeEnd - localNodeBegin;

    printf("[%lu] Nodes %lu to %lu\n", hostID, localNodeBegin, localNodeEnd);

    sendEdgeCounts(hostToNodes, localNumEdges, localEdges);
    std::atomic<uint64_t> edgesToReceive;
    edgesToReceive.store(receiveEdgeCounts());

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

    // TODO refactor out to a function

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

      // I won't check status here because there should be no reason why 
      // writing 8 bytes per write would fail.... (I hope at least)
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
    uint64_t nodeIndexOffset = headerSize + (localNodeBegin * sizeof(uint64_t));
    uint64_t nodesToWrite = localNumNodes;

    MPI_Status writeStatus;
    while (nodesToWrite != 0) {
      // make sure it writes all nodes (mpi might not; see status)....
      MPICheck(MPI_File_write_at(newGR, nodeIndexOffset, edgePrefixSum.data(),
                                 nodesToWrite, MPI_UINT64_T, &writeStatus));
      
      int itemsWritten;
      MPI_Get_count(&writeStatus, MPI_UINT64_T, &itemsWritten);
      nodesToWrite -= itemsWritten;
      nodeIndexOffset += itemsWritten * sizeof(uint64_t);
    }
    
    uint64_t edgeDestOffset = headerSize + (totalNumNodes * sizeof(uint64_t)) +
                              globalEdgeOffset * sizeof(uint32_t);

    // write edge dests
    for (unsigned i = 0; i < localNumNodes; i++) {
      std::vector<uint32_t> currentDests = localSrcToDest[i];
      uint64_t numToWrite = currentDests.size();

      while (numToWrite != 0) {
        MPICheck(MPI_File_write_at(newGR, edgeDestOffset, currentDests.data(),
                                   numToWrite, MPI_UINT32_T, &writeStatus));

        int itemsWritten;
        MPI_Get_count(&writeStatus, MPI_UINT32_T, &itemsWritten);
        numToWrite -= itemsWritten;
        edgeDestOffset += sizeof(uint32_t) * itemsWritten;
      }
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
