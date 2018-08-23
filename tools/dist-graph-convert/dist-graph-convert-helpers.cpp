/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "dist-graph-convert-helpers.h"

std::vector<uint32_t> readRandomNodeMapping(const std::string& nodeMapBinary,
                                            uint64_t nodeOffset,
                                            uint64_t numToRead) {
  MPI_File mb;
  MPICheck(MPI_File_open(MPI_COMM_WORLD, nodeMapBinary.c_str(), MPI_MODE_RDONLY,
                         MPI_INFO_NULL, &mb));

  uint64_t readPosition = nodeOffset * sizeof(uint32_t);
  uint64_t numRead      = 0;
  MPI_Status readStatus;
  std::vector<uint32_t> node2NewNode(numToRead);

  while (numToRead > 0) {
    // File_read can only go up to the max int
    uint64_t toLoad =
        std::min(numToRead, (uint64_t)std::numeric_limits<int>::max());
    MPI_File_read_at(mb, readPosition,
                     ((char*)(node2NewNode.data())) +
                         (numRead * sizeof(uint32_t)),
                     toLoad, MPI_UINT32_T, &readStatus);

    int nodesRead;
    MPI_Get_count(&readStatus, MPI_UINT32_T, &nodesRead);
    GALOIS_ASSERT(nodesRead != MPI_UNDEFINED, "Nodes read is MPI_UNDEFINED");
    numToRead -= nodesRead;
    numRead += nodesRead;
    readPosition += nodesRead * sizeof(uint32_t);
  }
  MPICheck(MPI_File_close(&mb));

  return node2NewNode;
}

void MPICheck(int errcode) {
  if (errcode != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, errcode);
  }
}

Uint64Pair readV1GrHeader(const std::string& grFile, bool isVoid) {
  MPI_File gr;
  MPICheck(MPI_File_open(MPI_COMM_WORLD, grFile.c_str(), MPI_MODE_RDONLY,
                         MPI_INFO_NULL, &gr));
  uint64_t grHeader[4];
  MPICheck(
      MPI_File_read_at(gr, 0, grHeader, 4, MPI_UINT64_T, MPI_STATUS_IGNORE));
  MPICheck(MPI_File_close(&gr));
  GALOIS_ASSERT(grHeader[0] == 1, "gr file must be version 1");

  if (!isVoid) {
    GALOIS_ASSERT(grHeader[1] != 0, "gr should have weights "
                                    "(specified in header)");
  }

  return Uint64Pair(grHeader[2], grHeader[3]);
}

std::vector<Uint64Pair> getHostToNodeMapping(uint64_t numHosts,
                                             uint64_t totalNumNodes) {
  GALOIS_ASSERT((totalNumNodes != 0), "host2node mapping needs numNodes");

  std::vector<Uint64Pair> hostToNodes;

  for (unsigned i = 0; i < numHosts; i++) {
    hostToNodes.emplace_back(
        galois::block_range((uint64_t)0, (uint64_t)totalNumNodes, i, numHosts));
  }

  return hostToNodes;
}

uint32_t findOwner(const uint64_t gID,
                   const std::vector<Uint64Pair>& ownerMapping) {
  uint32_t lb = 0;
  uint32_t ub = ownerMapping.size();

  while (lb < ub) {
    uint64_t mid      = lb + (ub - lb) / 2;
    auto& currentPair = ownerMapping[mid];

    if (gID >= currentPair.first && gID < currentPair.second) {
      return mid;
    } else if (gID < currentPair.first) {
      // MOVE DOWN
      ub = mid;
    } else if (gID >= currentPair.second) { // gid >= currentPair.second
      // MOVE UP
      lb = mid + 1;
    } else { // die; we should fall into one of the above cases
      GALOIS_DIE("Issue in findOwner in dist-graph-convert helpers.");
    }
  }

  // it should find something above...
  return -1;
}

uint64_t getFileSize(std::ifstream& openFile) {
  openFile.seekg(0, std::ios_base::end);
  return openFile.tellg();
}

Uint64Pair determineByteRange(std::ifstream& edgeListFile, uint64_t fileSize) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  uint64_t initialStart;
  uint64_t initialEnd;
  std::tie(initialStart, initialEnd) = galois::block_range(
      (uint64_t)0, (uint64_t)fileSize, hostID, totalNumHosts);

  // printf("[%lu] Initial byte %lu to %lu\n", hostID, initialStart,
  // initialEnd);

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

  return Uint64Pair(finalStart, finalEnd);
}

uint64_t accumulateValue(uint64_t localEdgeCount) {
  galois::DGAccumulator<uint64_t> accumulator;
  accumulator.reset();
  accumulator += localEdgeCount;
  return accumulator.reduce();
}

uint64_t findIndexPrefixSum(uint64_t targetWeight, uint64_t lb, uint64_t ub,
                            const std::vector<uint64_t>& prefixSum) {
  while (lb < ub) {
    uint64_t mid = lb + (ub - lb) / 2;
    uint64_t numUnits;

    if (mid != 0) {
      numUnits = prefixSum[mid - 1];
    } else {
      numUnits = 0;
    }

    if (numUnits <= targetWeight) {
      lb = mid + 1;
    } else {
      ub = mid;
    }
  }

  return lb;
}

Uint64Pair binSearchDivision(uint64_t id, uint64_t totalID,
                             const std::vector<uint64_t>& prefixSum) {
  uint64_t totalWeight        = prefixSum.back();
  uint64_t weightPerPartition = (totalWeight + totalID - 1) / totalID;
  uint64_t numThingsToSplit   = prefixSum.size();

  uint64_t lower;
  if (id != 0) {
    lower = findIndexPrefixSum(id * weightPerPartition, 0, numThingsToSplit,
                               prefixSum);
  } else {
    lower = 0;
  }
  uint64_t upper = findIndexPrefixSum((id + 1) * weightPerPartition, lower,
                                      numThingsToSplit, prefixSum);

  return Uint64Pair(lower, upper);
}

void findUniqueChunks(galois::DynamicBitSet& uniqueNodeBitset,
                      const std::vector<Uint64Pair>& chunkToNode,
                      galois::DynamicBitSet& uniqueChunkBitset) {
  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;
  printf("[%lu] Finding unique chunks\n", hostID);
  uniqueChunkBitset.reset();

  galois::do_all(galois::iterate((size_t)0, uniqueNodeBitset.size()),
                 [&](auto nodeIndex) {
                   if (uniqueNodeBitset.test(nodeIndex)) {
                     uniqueChunkBitset.set(findOwner(nodeIndex, chunkToNode));
                   }
                 },
                 galois::loopname("FindUniqueChunks"));

  freeVector(uniqueNodeBitset.get_vec());

  printf("[%lu] Unique chunks found\n", hostID);
}

void sendAndReceiveEdgeChunkCounts(std::vector<uint64_t>& chunkCounts) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Sending edge chunk counts\n", hostID);
  // send off my chunk count vector to others so all hosts can have the
  // same count of edges in a chunk
  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID)
      continue;
    galois::runtime::SendBuffer b;
    galois::runtime::gSerialize(b, chunkCounts);
    net.sendTagged(h, galois::runtime::evilPhase, b);
  }

  // receive chunk counts
  std::vector<uint64_t> recvChunkCounts;

  printf("[%lu] Receiving edge chunk counts\n", hostID);
  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID)
      continue;
    decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) rBuffer;

    do {
      rBuffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
    } while (!rBuffer);

    galois::runtime::gDeserialize(rBuffer->second, recvChunkCounts);

    for (unsigned i = 0; i < chunkCounts.size(); i++) {
      chunkCounts[i] += recvChunkCounts[i];
    }
  }
  galois::runtime::evilPhase++;
}

std::vector<Uint64Pair>
getChunkToHostMapping(const std::vector<uint64_t>& chunkCountsPrefixSum,
                      const std::vector<Uint64Pair>& chunkToNode) {
  std::vector<Uint64Pair> finalMapping;

  uint64_t hostID        = galois::runtime::getSystemNetworkInterface().ID;
  uint64_t totalNumHosts = galois::runtime::getSystemNetworkInterface().Num;
  for (uint64_t h = 0; h < totalNumHosts; h++) {
    uint64_t lowerChunk;
    uint64_t upperChunk;

    // get the lower/upper chunk assigned to host h
    std::tie(lowerChunk, upperChunk) =
        binSearchDivision(h, totalNumHosts, chunkCountsPrefixSum);

    uint64_t lowerNode = chunkToNode[lowerChunk].first;
    uint64_t upperNode = chunkToNode[upperChunk].first;

    if (hostID == 0) {
      uint64_t edgeCount;
      if (lowerChunk == upperChunk) {
        edgeCount = 0;
      } else if (lowerChunk == 0) {
        edgeCount = chunkCountsPrefixSum[upperChunk - 1];
      } else {
        edgeCount = chunkCountsPrefixSum[upperChunk - 1] -
                    chunkCountsPrefixSum[lowerChunk - 1];
      }
      printf("Host %lu gets nodes %lu to %lu (count %lu), with %lu edges\n", h,
             lowerNode, upperNode, upperNode - lowerNode, edgeCount);
    }

    finalMapping.emplace_back(Uint64Pair(lowerNode, upperNode));
  }

  return finalMapping;
}

std::vector<Uint64Pair>
getChunkToHostMappingLinear(const std::vector<uint64_t>& chunkCountsPrefixSum,
                            const std::vector<Uint64Pair>& chunkToNode) {
  GALOIS_DIE("Currently does not work."); // TODO fix this function
  uint64_t hostID        = galois::runtime::getSystemNetworkInterface().ID;
  uint64_t totalNumHosts = galois::runtime::getSystemNetworkInterface().Num;

  uint64_t totalWeight = chunkCountsPrefixSum.back();
  uint64_t weightPerPartition =
      (totalWeight + totalNumHosts - 1) / totalNumHosts;
  uint64_t totalChunks = chunkCountsPrefixSum.size();

  // TODO corner case handling (1 host, more hosts than chunks)

  uint32_t currentHost  = 0;
  uint64_t currentChunk = 0;

  uint64_t accountedEdges = 0;

  std::vector<uint64_t> hostRanges(totalNumHosts + 1);
  hostRanges[0] = 0;

  while (currentChunk < totalChunks && currentHost < totalNumHosts) {
    uint32_t hostsRemaining = totalNumHosts - currentHost;
    GALOIS_ASSERT(totalChunks - currentChunk >= hostsRemaining);

    // Handle case where only 1 host left or 1-1 host-chunk mapping
    if (hostsRemaining == 1) {
      // assign the rest of chunks to last host
      hostRanges[currentHost + 1] = totalChunks;
    } else if ((totalChunks - currentChunk) == hostsRemaining) {
      // one chunk to each host
      for (unsigned i = 0; i < hostsRemaining; i++) {
        hostRanges[++currentHost] = (++currentChunk);
      }
    }

    // Number of edges in current chunk
    uint64_t chunkEdges;
    if (currentChunk > 0) {
      chunkEdges = chunkCountsPrefixSum[currentChunk] -
                   chunkCountsPrefixSum[currentChunk - 1];
    } else { // currentChunk == 0
      chunkEdges = chunkCountsPrefixSum[0];
    }

    // Num edges division currently has not accounting chunkEdges into it
    uint64_t edgeCountWithoutCurrent;
    if (currentChunk > 0) {
      edgeCountWithoutCurrent =
          chunkCountsPrefixSum[currentChunk] - accountedEdges - chunkEdges;
    } else { // currentChunk == 0
      edgeCountWithoutCurrent = 0;
    }

    // If chunk edges is large, then don't add to current host unless host
    // doesn't have much to begin with
    if (chunkEdges > (3 * weightPerPartition / 4)) {
      // If adding this chunk to current host too much, add to next host
      // instead
      if (edgeCountWithoutCurrent > (weightPerPartition / 4)) {
        GALOIS_ASSERT(currentChunk != 0);

        // assign to next host
        // Beginning of next is current chunk
        hostRanges[currentHost + 1] = currentChunk;
        accountedEdges              = chunkCountsPrefixSum[currentChunk - 1];
        currentHost++;
        continue;
      }
    }

    // otherwise handle regularly
    uint64_t currentEdgeCount = edgeCountWithoutCurrent + chunkEdges;

    if (currentEdgeCount >= weightPerPartition) {
      hostRanges[++currentHost] = currentChunk + 1;
      accountedEdges            = chunkCountsPrefixSum[currentChunk];
    }

    currentChunk++;
  }

  galois::gPrint("[", hostID, "] Done here\n");

  GALOIS_ASSERT(hostRanges[0] == 0);
  GALOIS_ASSERT(hostRanges[totalNumHosts] == totalChunks);

  // handle pair creation
  std::vector<Uint64Pair> finalMapping;
  for (uint64_t h = 0; h < totalNumHosts; h++) {
    uint64_t lowerChunk = hostRanges[h];
    uint64_t upperChunk = hostRanges[h + 1];

    uint64_t lowerNode = chunkToNode[lowerChunk].first;
    uint64_t upperNode = chunkToNode[upperChunk].first;

    if (hostID == 0) {
      uint64_t edgeCount;
      if (lowerChunk == upperChunk) {
        edgeCount = 0;
      } else if (lowerChunk == 0) {
        edgeCount = chunkCountsPrefixSum[upperChunk - 1];
      } else {
        edgeCount = chunkCountsPrefixSum[upperChunk - 1] -
                    chunkCountsPrefixSum[lowerChunk - 1];
      }
      printf("Host %lu gets nodes %lu to %lu (count %lu), with %lu edges\n", h,
             lowerNode, upperNode, upperNode - lowerNode, edgeCount);
    }

    finalMapping.emplace_back(Uint64Pair(lowerNode, upperNode));
  }

  return finalMapping;
}

DoubleUint64Pair getNodesToReadFromGr(const std::string& inputGr) {
  uint32_t hostID        = galois::runtime::getSystemNetworkInterface().ID;
  uint32_t totalNumHosts = galois::runtime::getSystemNetworkInterface().Num;

  galois::graphs::OfflineGraph offlineGr(inputGr);
  auto nodeAndEdgeRange = offlineGr.divideByNode(0, 1, hostID, totalNumHosts);
  auto& nodeRange       = nodeAndEdgeRange.first;
  auto& edgeRange       = nodeAndEdgeRange.second;
  Uint64Pair nodePair(*nodeRange.first, *nodeRange.second);
  Uint64Pair edgePair(*edgeRange.first, *edgeRange.second);
  return DoubleUint64Pair(nodePair, edgePair);
}

std::vector<uint32_t> loadCleanEdgesFromBufferedGraph(
    const std::string& inputFile, Uint64Pair nodesToRead,
    Uint64Pair edgesToRead, uint64_t totalNumNodes, uint64_t totalNumEdges) {
  galois::graphs::BufferedGraph<void> bufGraph;
  bufGraph.loadPartialGraph(inputFile, nodesToRead.first, nodesToRead.second,
                            edgesToRead.first, edgesToRead.second,
                            totalNumNodes, totalNumEdges);
  size_t numNodesToRead = nodesToRead.second - nodesToRead.first;
  std::vector<std::set<uint32_t>> nonDupSets(numNodesToRead);

  // insert edge destinations of each node into a set (i.e. no duplicates)
  galois::do_all(galois::iterate(nodesToRead.first, nodesToRead.second),
                 [&](uint32_t gID) {
                   size_t vectorIndex = gID - nodesToRead.first;

                   uint64_t edgeBegin = *bufGraph.edgeBegin(gID);
                   uint64_t edgeEnd   = *bufGraph.edgeEnd(gID);

                   for (uint64_t i = edgeBegin; i < edgeEnd; i++) {
                     uint32_t edgeDest = bufGraph.edgeDestination(i);
                     if (edgeDest != gID) {
                       nonDupSets[vectorIndex].insert(edgeDest);
                     }
                   }
                 },
                 galois::steal(), galois::loopname("FindCleanEdges"));

  // get total num edges remaining
  uint64_t edgesRemaining = 0;
  for (unsigned i = 0; i < numNodesToRead; i++) {
    edgesRemaining += nonDupSets[i].size();
  }

  std::vector<uint32_t> edgeData(edgesRemaining * 2);

  uint64_t counter = 0;

  // (serially) create the edge vector; TODO it's possible to parallelize
  // this loop using a prefix sum of edges....; worth doing?
  for (unsigned i = 0; i < numNodesToRead; i++) {
    std::set<uint32_t> currentSet = nonDupSets[i];
    uint32_t currentGID           = i + nodesToRead.first;

    for (auto dest : currentSet) {
      edgeData[counter * 2]     = currentGID; // src
      edgeData[counter * 2 + 1] = dest;
      counter++;
    }
  }

  return edgeData;
}

uint64_t receiveEdgeCounts() {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Receiving edge counts\n", hostID);

  uint64_t edgesToReceive = 0;

  // receive
  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID)
      continue;
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

void receiveAssignedEdges(std::atomic<uint64_t>& edgesToReceive,
                          const std::vector<Uint64Pair>& hostToNodes,
                          std::vector<std::vector<uint32_t>>& localSrcToDest,
                          std::vector<std::vector<uint32_t>>& localSrcToData,
                          std::vector<std::mutex>& nodeLocks) {
  auto& net       = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;

  printf("[%lu] Going to receive assigned edges\n", hostID);

  // receive edges
  galois::on_each(
      [&](unsigned tid, unsigned nthreads) {
        std::vector<uint32_t> recvVector;
        std::vector<uint32_t> recvDataVector;

        while (edgesToReceive) {
          decltype(
              net.recieveTagged(galois::runtime::evilPhase, nullptr)) rBuffer;
          rBuffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);

          // the buffer will have edge data as well if localsrctodata is
          // nonempty (it will be nonempty if initialized to non-empty by the
          // send function, and the send function only initializes it if it is
          // going to send edge data
          if (rBuffer) {
            auto& receiveBuffer = rBuffer->second;
            while (receiveBuffer.r_size() > 0) {
              uint64_t src;
              if (localSrcToData.empty()) {
                // receive only dest data
                galois::runtime::gDeserialize(receiveBuffer, src, recvVector);
              } else {
                // receive edge data as well
                galois::runtime::gDeserialize(receiveBuffer, src, recvVector,
                                              recvDataVector);
              }

              edgesToReceive -= recvVector.size();
              GALOIS_ASSERT(findOwner(src, hostToNodes) == hostID);
              uint32_t localID = src - hostToNodes[hostID].first;

              nodeLocks[localID].lock();
              if (localSrcToData.empty()) {
                // deal with only destinations
                for (unsigned i = 0; i < recvVector.size(); i++) {
                  localSrcToDest[localID].emplace_back(recvVector[i]);
                }
              } else {
                // deal with destinations and data
                for (unsigned i = 0; i < recvVector.size(); i++) {
                  localSrcToDest[localID].emplace_back(recvVector[i]);
                  localSrcToData[localID].emplace_back(recvDataVector[i]);
                }
              }
              nodeLocks[localID].unlock();
            }
          }
        }
      },
      galois::loopname("EdgeReceiving"));
  galois::runtime::evilPhase++;

  printf("[%lu] Receive assigned edges finished\n", hostID);
}

std::vector<uint64_t> getEdgesPerHost(uint64_t localAssignedEdges) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Informing other hosts about number of edges\n", hostID);

  std::vector<uint64_t> edgesPerHost(totalNumHosts);

  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID)
      continue;
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

std::vector<uint32_t>
flattenVectors(std::vector<std::vector<uint32_t>>& vectorOfVectors) {
  std::vector<uint32_t> finalVector;
  uint64_t vectorsToFlatten = vectorOfVectors.size();

  for (unsigned i = 0; i < vectorsToFlatten; i++) {
    auto& curVector = vectorOfVectors[i];
    finalVector.insert(finalVector.end(), curVector.begin(), curVector.end());
    // free the memory up
    freeVector(vectorOfVectors[i]);
  }

  return finalVector;
}

void writeGrHeader(MPI_File& gr, uint64_t version, uint64_t sizeOfEdge,
                   uint64_t totalNumNodes, uint64_t totalEdgeCount) {
  // I won't check status here because there should be no reason why
  // writing 8 bytes per write would fail.... (I hope at least)
  MPICheck(
      MPI_File_write_at(gr, 0, &version, 1, MPI_UINT64_T, MPI_STATUS_IGNORE));
  MPICheck(MPI_File_write_at(gr, sizeof(uint64_t), &sizeOfEdge, 1, MPI_UINT64_T,
                             MPI_STATUS_IGNORE));
  MPICheck(MPI_File_write_at(gr, sizeof(uint64_t) * 2, &totalNumNodes, 1,
                             MPI_UINT64_T, MPI_STATUS_IGNORE));
  MPICheck(MPI_File_write_at(gr, sizeof(uint64_t) * 3, &totalEdgeCount, 1,
                             MPI_UINT64_T, MPI_STATUS_IGNORE));
}

void writeNodeIndexData(MPI_File& gr, uint64_t nodesToWrite,
                        uint64_t nodeIndexOffset,
                        const std::vector<uint64_t>& edgePrefixSum) {
  MPI_Status writeStatus;
  uint64_t totalWritten = 0;
  while (nodesToWrite != 0) {
    uint64_t toWrite =
        std::min(nodesToWrite, (uint64_t)std::numeric_limits<int>::max());

    MPICheck(MPI_File_write_at(gr, nodeIndexOffset,
                               ((uint64_t*)edgePrefixSum.data()) + totalWritten,
                               toWrite, MPI_UINT64_T, &writeStatus));

    int itemsWritten;
    MPI_Get_count(&writeStatus, MPI_UINT64_T, &itemsWritten);
    GALOIS_ASSERT(itemsWritten != MPI_UNDEFINED,
                  "itemsWritten is MPI_UNDEFINED");
    nodesToWrite -= itemsWritten;
    totalWritten += itemsWritten;
    nodeIndexOffset += itemsWritten * sizeof(uint64_t);
  }
}

// vector of vectors version
void writeEdgeDestData(MPI_File& gr, uint64_t edgeDestOffset,
                       std::vector<std::vector<uint32_t>>& localSrcToDest) {
  MPI_Status writeStatus;

  for (unsigned i = 0; i < localSrcToDest.size(); i++) {
    std::vector<uint32_t> currentDests = localSrcToDest[i];
    uint64_t numToWrite                = currentDests.size();
    uint64_t totalWritten              = 0;

    while (numToWrite != 0) {
      uint64_t toWrite =
          std::min(numToWrite, (uint64_t)std::numeric_limits<int>::max());

      MPICheck(MPI_File_write_at(
          gr, edgeDestOffset, ((uint32_t*)currentDests.data()) + totalWritten,
          toWrite, MPI_UINT32_T, &writeStatus));

      int itemsWritten;
      MPI_Get_count(&writeStatus, MPI_UINT32_T, &itemsWritten);
      GALOIS_ASSERT(itemsWritten != MPI_UNDEFINED,
                    "itemsWritten is MPI_UNDEFINED");
      numToWrite -= itemsWritten;
      totalWritten += itemsWritten;
      edgeDestOffset += sizeof(uint32_t) * itemsWritten;
    }
  }
}

// 1 vector version (MUCH FASTER, USE WHEN POSSIBLE)
void writeEdgeDestData(MPI_File& gr, uint64_t edgeDestOffset,
                       std::vector<uint32_t>& destVector) {
  MPI_Status writeStatus;
  uint64_t numToWrite   = destVector.size();
  uint64_t totalWritten = 0;

  while (numToWrite != 0) {
    uint64_t toWrite =
        std::min(numToWrite, (uint64_t)std::numeric_limits<int>::max());

    MPICheck(MPI_File_write_at(gr, edgeDestOffset,
                               ((uint32_t*)destVector.data()) + totalWritten,
                               toWrite, MPI_UINT32_T, &writeStatus));

    int itemsWritten;
    MPI_Get_count(&writeStatus, MPI_UINT32_T, &itemsWritten);
    GALOIS_ASSERT(itemsWritten != MPI_UNDEFINED,
                  "itemsWritten is MPI_UNDEFINED");
    numToWrite -= itemsWritten;
    totalWritten += itemsWritten;
    edgeDestOffset += sizeof(uint32_t) * itemsWritten;
  }
}

void writeEdgeDataData(MPI_File& gr, uint64_t edgeDataOffset,
                       const std::vector<uint32_t>& edgeDataToWrite) {
  MPI_Status writeStatus;
  uint64_t numToWrite = edgeDataToWrite.size();
  uint64_t numWritten = 0;

  while (numToWrite != 0) {
    uint64_t toWrite =
        std::min(numToWrite, (uint64_t)std::numeric_limits<int>::max());

    MPICheck(MPI_File_write_at(gr, edgeDataOffset,
                               ((uint32_t*)edgeDataToWrite.data()) + numWritten,
                               toWrite, MPI_UINT32_T, &writeStatus));
    int itemsWritten;
    MPI_Get_count(&writeStatus, MPI_UINT32_T, &itemsWritten);
    GALOIS_ASSERT(itemsWritten != MPI_UNDEFINED,
                  "itemsWritten is MPI_UNDEFINED");
    numToWrite -= itemsWritten;
    numWritten += itemsWritten;
    edgeDataOffset += itemsWritten * sizeof(uint32_t);
  }
}

void writeToGr(const std::string& outputFile, uint64_t totalNumNodes,
               uint64_t totalNumEdges, uint64_t localNumNodes,
               uint64_t localNodeBegin, uint64_t globalEdgeOffset,
               std::vector<std::vector<uint32_t>>& localSrcToDest,
               std::vector<std::vector<uint32_t>>& localSrcToData) {
  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;

  printf("[%lu] Beginning write to file\n", hostID);
  MPI_File newGR;
  MPICheck(MPI_File_open(MPI_COMM_WORLD, outputFile.c_str(),
                         MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                         &newGR));

  if (hostID == 0) {
    if (localSrcToData.empty()) {
      writeGrHeader(newGR, 1, 0, totalNumNodes, totalNumEdges);
    } else {
      // edge data size hard set to 4 if there is data to write (uint32_t)
      writeGrHeader(newGR, 1, 4, totalNumNodes, totalNumEdges);
    }
  }

  if (localNumNodes > 0) {
    // prepare edge prefix sum for file writing
    std::vector<uint64_t> edgePrefixSum(localNumNodes);
    edgePrefixSum[0] = localSrcToDest[0].size();
    for (unsigned i = 1; i < localNumNodes; i++) {
      edgePrefixSum[i] = (edgePrefixSum[i - 1] + localSrcToDest[i].size());
    }

    // account for edge offset
    for (unsigned i = 0; i < localNumNodes; i++) {
      edgePrefixSum[i] = edgePrefixSum[i] + globalEdgeOffset;
    }

    // begin file writing
    uint64_t headerSize      = sizeof(uint64_t) * 4;
    uint64_t nodeIndexOffset = headerSize + (localNodeBegin * sizeof(uint64_t));
    printf("[%lu] Write node index data\n", hostID);
    writeNodeIndexData(newGR, localNumNodes, nodeIndexOffset, edgePrefixSum);
    freeVector(edgePrefixSum);

    uint64_t edgeDestOffset = headerSize + (totalNumNodes * sizeof(uint64_t)) +
                              globalEdgeOffset * sizeof(uint32_t);
    printf("[%lu] Write edge dest data\n", hostID);
    std::vector<uint32_t> destVector = flattenVectors(localSrcToDest);
    freeVector(localSrcToDest);
    writeEdgeDestData(newGR, edgeDestOffset, destVector);

    // edge data writing if necessary
    if (!localSrcToData.empty()) {
      uint64_t edgeDataOffset = getOffsetToLocalEdgeData(
          totalNumNodes, totalNumEdges, globalEdgeOffset);
      printf("[%lu] Write edge data data\n", hostID);
      std::vector<uint32_t> dataVector = flattenVectors(localSrcToData);
      freeVector(localSrcToData);
      writeEdgeDataData(newGR, edgeDataOffset, dataVector);
    }

    printf("[%lu] Write to file done\n", hostID);
  }

  MPICheck(MPI_File_close(&newGR));
}

void writeToLux(const std::string& outputFile, uint64_t totalNumNodes,
                uint64_t totalNumEdges, uint64_t localNumNodes,
                uint64_t localNodeBegin, uint64_t globalEdgeOffset,
                std::vector<std::vector<uint32_t>>& localSrcToDest,
                std::vector<std::vector<uint32_t>>& localSrcToData) {
  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;

  printf("[%lu] Beginning write to file\n", hostID);
  MPI_File newGR;
  MPICheck(MPI_File_open(MPI_COMM_WORLD, outputFile.c_str(),
                         MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                         &newGR));

  // Lux header
  if (hostID == 0) {
    // cast down the node data
    uint32_t castDown = totalNumNodes;
    MPICheck(MPI_File_write_at(newGR, 0, &castDown, 1, MPI_UINT32_T,
             MPI_STATUS_IGNORE));
    MPICheck(MPI_File_write_at(newGR, sizeof(uint32_t), &totalNumEdges, 1,
                               MPI_UINT64_T, MPI_STATUS_IGNORE));
  }

  if (localNumNodes > 0) {
    // prepare edge prefix sum for file writing
    std::vector<uint64_t> edgePrefixSum(localNumNodes);
    edgePrefixSum[0] = localSrcToDest[0].size();
    for (unsigned i = 1; i < localNumNodes; i++) {
      edgePrefixSum[i] = (edgePrefixSum[i - 1] + localSrcToDest[i].size());
    }

    // account for edge offset
    for (unsigned i = 0; i < localNumNodes; i++) {
      edgePrefixSum[i] = edgePrefixSum[i] + globalEdgeOffset;
    }

    // begin file writing
    // Lux header differs from Galois header
    uint64_t headerSize      = sizeof(uint32_t) + sizeof(uint64_t);
    uint64_t nodeIndexOffset = headerSize + (localNodeBegin * sizeof(uint64_t));

    printf("[%lu] Write node index data\n", hostID);
    writeNodeIndexData(newGR, localNumNodes, nodeIndexOffset, edgePrefixSum);
    freeVector(edgePrefixSum);

    uint64_t edgeDestOffset = headerSize + (totalNumNodes * sizeof(uint64_t)) +
                              globalEdgeOffset * sizeof(uint32_t);
    printf("[%lu] Write edge dest data\n", hostID);
    std::vector<uint32_t> destVector = flattenVectors(localSrcToDest);
    freeVector(localSrcToDest);
    writeEdgeDestData(newGR, edgeDestOffset, destVector);

    // edge data writing if necessary
    if (!localSrcToData.empty()) {
      uint64_t byteOffsetToEdgeData = sizeof(uint32_t) + sizeof(uint64_t) +
                                      (totalNumNodes * sizeof(uint64_t)) +
                                      (totalNumEdges * sizeof(uint32_t));
      byteOffsetToEdgeData += globalEdgeOffset * sizeof(uint32_t);
      // NO PADDING
      uint64_t edgeDataOffset = byteOffsetToEdgeData;

      printf("[%lu] Write edge data data\n", hostID);
      std::vector<uint32_t> dataVector = flattenVectors(localSrcToData);
      freeVector(localSrcToData);
      writeEdgeDataData(newGR, edgeDataOffset, dataVector);
    }

    printf("[%lu] Write to file done\n", hostID);
  }

  MPICheck(MPI_File_close(&newGR));
}

std::vector<uint32_t> generateRandomNumbers(uint64_t count, uint64_t seed,
                                            uint64_t lower, uint64_t upper) {
  std::minstd_rand0 rGenerator;
  rGenerator.seed(seed);
  std::uniform_int_distribution<uint32_t> rDist(lower, upper);

  std::vector<uint32_t> randomNumbers;
  randomNumbers.reserve(count);
  for (unsigned i = 0; i < count; i++) {
    randomNumbers.emplace_back(rDist(rGenerator));
  }

  return randomNumbers;
}

uint64_t getOffsetToLocalEdgeData(uint64_t totalNumNodes,
                                  uint64_t totalNumEdges,
                                  uint64_t localEdgeBegin) {
  uint64_t byteOffsetToEdgeData = (4 * sizeof(uint64_t)) +             // header
                                  (totalNumNodes * sizeof(uint64_t)) + // nodes
                                  (totalNumEdges * sizeof(uint32_t));  // edges
  // version 1: determine if padding is necessary at end of file +
  // add it (64 byte alignment since edges are 32 bytes in version 1)
  if (totalNumEdges % 2) {
    byteOffsetToEdgeData += sizeof(uint32_t);
  }
  byteOffsetToEdgeData += localEdgeBegin * sizeof(uint32_t);

  return byteOffsetToEdgeData;
}

Uint64Pair getLocalAssignment(uint64_t numToSplit) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  return galois::block_range((uint64_t)0, numToSplit, hostID, totalNumHosts);
}
