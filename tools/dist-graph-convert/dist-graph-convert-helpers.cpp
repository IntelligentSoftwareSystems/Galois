/** Distributed graph converter helpers -*- C++ -*-
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
 * Distributed graph converter helper implementations.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#include "dist-graph-convert-helpers.h"

void MPICheck(int errcode) {
  if (errcode != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, errcode);
  }
}

std::vector<std::pair<uint64_t, uint64_t>> getHostToNodeMapping(
    uint64_t numHosts, uint64_t totalNumNodes
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

uint32_t findOwner(const uint64_t gID, 
            const std::vector<std::pair<uint64_t, uint64_t>>& ownerMapping) {
  uint32_t lb = 0;
  uint32_t ub = ownerMapping.size();

  while (lb < ub) {
    uint64_t mid = lb + (ub - lb) / 2;
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

std::pair<uint64_t, uint64_t> determineByteRange(std::ifstream& edgeListFile,
                                                 uint64_t fileSize) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  uint64_t initialStart;
  uint64_t initialEnd;
  std::tie(initialStart, initialEnd) = galois::block_range((uint64_t)0, 
                                                           (uint64_t)fileSize,
                                                           hostID, 
                                                           totalNumHosts);

  //printf("[%lu] Initial byte %lu to %lu\n", hostID, initialStart, initialEnd);

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

std::pair<uint64_t, uint64_t> binSearchDivision(uint64_t id, uint64_t totalID, 
                                  const std::vector<uint64_t>& prefixSum) {
  uint64_t totalWeight = prefixSum.back();
  uint64_t weightPerPartition = (totalWeight + totalID - 1) / totalID;
  uint64_t numThingsToSplit = prefixSum.size();

  uint64_t lower;
  if (id != 0) {
    lower = findIndexPrefixSum(id * weightPerPartition, 0, numThingsToSplit,
                               prefixSum);
  } else {
    lower = 0;
  }
  uint64_t upper = findIndexPrefixSum((id + 1) * weightPerPartition, 
                                      lower, numThingsToSplit, prefixSum);
  
  return std::pair<uint64_t, uint64_t>(lower, upper);
}


std::set<uint64_t> 
findUniqueChunks(const std::set<uint64_t>& uniqueNodes,
                const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode) {
  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;

  printf("[%lu] Finding unique chunks\n", hostID);
  galois::substrate::PerThreadStorage<std::set<uint64_t>> threadUniqueChunks;

  galois::do_all(
    galois::iterate(uniqueNodes.cbegin(), uniqueNodes.cend()),
    [&] (auto uniqueNode) {
      std::set<uint64_t>& localSet = *threadUniqueChunks.getLocal();
      localSet.insert(findOwner(uniqueNode, chunkToNode));
    },
    galois::loopname("FindUniqueChunks"),
    galois::no_stats(),
    galois::steal<false>(),
    galois::timeit()
  );

  std::set<uint64_t> uniqueChunks;

  for (unsigned i = 0; i < threadUniqueChunks.size(); i++) {
    auto& tSet = *threadUniqueChunks.getRemote(i);
    for (auto chunkID : tSet) {
      uniqueChunks.insert(chunkID);
    }
  }

  printf("[%lu] Have %lu unique chunk(s)\n", hostID, uniqueChunks.size());

  return uniqueChunks;
}

void sendAndReceiveEdgeChunkCounts(std::vector<uint64_t>& chunkCounts) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Sending edge chunk counts\n", hostID);
  // send off my chunk count vector to others so all hosts can have the
  // same count of edges in a chunk
  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID) continue;
    galois::runtime::SendBuffer b;
    galois::runtime::gSerialize(b, chunkCounts);
    net.sendTagged(h, galois::runtime::evilPhase, b);
  }

  // receive chunk counts
  std::vector<uint64_t> recvChunkCounts;

  printf("[%lu] Receiving edge chunk counts\n", hostID);
  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID) continue;
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

std::vector<std::pair<uint64_t, uint64_t>> getChunkToHostMapping(
      const std::vector<uint64_t>& chunkCountsPrefixSum,
      const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode) {
  std::vector<std::pair<uint64_t, uint64_t>> finalMapping;

  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;
  uint64_t totalNumHosts = galois::runtime::getSystemNetworkInterface().Num;
  for (uint64_t h = 0; h < totalNumHosts; h++) {
    uint64_t lowerChunk;
    uint64_t upperChunk;

    // get the lower/upper chunk assigned to host h
    std::tie(lowerChunk, upperChunk) = binSearchDivision(h, totalNumHosts, 
                                                         chunkCountsPrefixSum);
    
    uint64_t lowerNode = chunkToNode[lowerChunk].first;
    uint64_t upperNode = chunkToNode[upperChunk].first;

    if (hostID == 0) {
      printf("Host %lu gets nodes %lu to %lu (count %lu)\n", h, 
              lowerNode, upperNode, upperNode - lowerNode);
    }

    finalMapping.emplace_back(std::pair<uint64_t, uint64_t>(lowerNode, 
                                                            upperNode));
  }

  return finalMapping;
}

DoubleUint64Pair getNodesToReadFromGr(const std::string& inputGr) {
  uint32_t hostID = galois::runtime::getSystemNetworkInterface().ID;
  uint32_t totalNumHosts = galois::runtime::getSystemNetworkInterface().Num;

  galois::graphs::OfflineGraph offlineGr(inputGr);
  auto nodeAndEdgeRange = offlineGr.divideByNode(0, 1, hostID, totalNumHosts);
  auto& nodeRange = nodeAndEdgeRange.first;
  auto& edgeRange = nodeAndEdgeRange.second;
  Uint64Pair nodePair(*nodeRange.first, *nodeRange.second);
  Uint64Pair edgePair(*edgeRange.first, *edgeRange.second);
  return DoubleUint64Pair(nodePair, edgePair);
}

std::vector<uint32_t> loadTransposedEdgesFromMPIGraph(
    const std::string& inputFile, Uint64Pair nodesToRead, 
    Uint64Pair edgesToRead, uint64_t totalNumNodes, uint64_t totalNumEdges
) {
  galois::graphs::MPIGraph<uint32_t> mpiGraph;
  mpiGraph.loadPartialGraph(inputFile, nodesToRead.first, nodesToRead.second,
                            edgesToRead.first, edgesToRead.second, 
                            totalNumNodes, totalNumEdges);

  std::vector<uint32_t> edgeData((edgesToRead.second - edgesToRead.first) * 3);

  if (edgeData.size() > 0) {
    galois::do_all(
      galois::iterate(nodesToRead.first, nodesToRead.second),
      [&] (uint32_t gID) {
        uint64_t edgeBegin = *mpiGraph.edgeBegin(gID);
        uint64_t edgeEnd = *mpiGraph.edgeEnd(gID);

        // offset into which we should start writing data in edgeData
        uint64_t edgeDataOffset = (edgeBegin - edgesToRead.first) * 3;
        
        // loop through all edges, save data
        for (uint64_t i = edgeBegin; i < edgeEnd; i++) {
          uint32_t edgeSource = mpiGraph.edgeDestination(i);
          uint32_t edgeWeight = mpiGraph.edgeData(i);

          // note that src is saved as dest and dest is aved as source 
          // (transpose)
          edgeData[edgeDataOffset] = edgeSource;
          edgeData[edgeDataOffset + 1] = gID;
          edgeData[edgeDataOffset + 2] = edgeWeight;
          edgeDataOffset += 3;
        }
      },
      galois::loopname("LoadTransposeEdgesMPIGraph"),
      galois::timeit(),
      galois::no_stats()
    );
  }
  
  return edgeData;
}

uint64_t receiveEdgeCounts() {
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Receiving edge counts\n", hostID);

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

void receiveAssignedEdges(std::atomic<uint64_t>& edgesToReceive,
    const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
    std::vector<std::vector<uint32_t>>& localSrcToDest,
    std::vector<std::vector<uint32_t>>& localSrcToData,
    std::vector<std::mutex>& nodeLocks)
{
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;

  printf("[%lu] Going to receive assigned edges\n", hostID);

  // receive edges
  galois::on_each(
    [&] (unsigned tid, unsigned nthreads) {
      std::vector<uint32_t> recvVector;
      std::vector<uint32_t> recvDataVector;

      while (edgesToReceive) {
        decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) 
            rBuffer;
        rBuffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        
        // the buffer will have edge data as well if localsrctodata is nonempty
        // (it will be nonempty if initialized to non-empty by the send 
        // function, and the send function only initializes it if it is going
        // to send edge data
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
    galois::loopname("EdgeReceiving"),
    galois::timeit(),
    galois::no_stats()
  );
  galois::runtime::evilPhase++; 

  printf("[%lu] Receive assigned edges finished\n", hostID);
}

std::vector<uint64_t> getEdgesPerHost(uint64_t localAssignedEdges) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Informing other hosts about number of edges\n", hostID);

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
  MPICheck(MPI_File_write_at(gr, 0, &version, 1, MPI_UINT64_T, 
           MPI_STATUS_IGNORE));
  MPICheck(MPI_File_write_at(gr, sizeof(uint64_t), &sizeOfEdge, 1, 
                             MPI_UINT64_T, MPI_STATUS_IGNORE));
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
    MPICheck(MPI_File_write_at(gr, nodeIndexOffset, 
                               ((uint64_t*)edgePrefixSum.data()) + totalWritten,
                               nodesToWrite, MPI_UINT64_T, &writeStatus));
    
    int itemsWritten;
    MPI_Get_count(&writeStatus, MPI_UINT64_T, &itemsWritten);
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
    uint64_t numToWrite = currentDests.size();
    uint64_t totalWritten = 0;

    while (numToWrite != 0) {
      MPICheck(MPI_File_write_at(gr, edgeDestOffset, 
                                ((uint32_t*)currentDests.data()) + totalWritten,
                                numToWrite, MPI_UINT32_T, &writeStatus));

      int itemsWritten;
      MPI_Get_count(&writeStatus, MPI_UINT32_T, &itemsWritten);
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
  uint64_t numToWrite = destVector.size();
  uint64_t totalWritten = 0;

  while (numToWrite != 0) {
    MPICheck(MPI_File_write_at(gr, edgeDestOffset, 
                              ((uint32_t*)destVector.data()) + totalWritten,
                              numToWrite, MPI_UINT32_T, &writeStatus));

    int itemsWritten;
    MPI_Get_count(&writeStatus, MPI_UINT32_T, &itemsWritten);
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
    MPICheck(MPI_File_write_at(gr, edgeDataOffset, 
                             ((uint32_t*)edgeDataToWrite.data()) + numWritten,
                             numToWrite, MPI_UINT32_T, &writeStatus));
    int itemsWritten;
    MPI_Get_count(&writeStatus, MPI_UINT32_T, &itemsWritten);
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
           MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &newGR));

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
    uint64_t headerSize = sizeof(uint64_t) * 4;
    uint64_t nodeIndexOffset = headerSize + 
                               (localNodeBegin * sizeof(uint64_t));
    printf("[%lu] Write node index data\n", hostID);
    writeNodeIndexData(newGR, localNumNodes, nodeIndexOffset, edgePrefixSum);
    freeVector(edgePrefixSum);

    uint64_t edgeDestOffset = headerSize + 
                              (totalNumNodes * sizeof(uint64_t)) +
                              globalEdgeOffset * sizeof(uint32_t);
    printf("[%lu] Write edge dest data\n", hostID);
    std::vector<uint32_t> destVector = flattenVectors(localSrcToDest);
    freeVector(localSrcToDest);
    writeEdgeDestData(newGR, edgeDestOffset, destVector);

    // edge data writing if necessary
    if (!localSrcToData.empty()) {
      uint64_t edgeDataOffset = getOffsetToLocalEdgeData(totalNumNodes, 
                                                         totalNumEdges,
                                                         globalEdgeOffset);
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
  uint64_t byteOffsetToEdgeData = (4 * sizeof(uint64_t)) + // header
                                  (totalNumNodes * sizeof(uint64_t)) + // nodes
                                  (totalNumEdges * sizeof(uint32_t)); // edges
  // version 1: determine if padding is necessary at end of file +
  // add it (64 byte alignment since edges are 32 bytes in version 1)
  if (totalNumEdges % 2) {
    byteOffsetToEdgeData += sizeof(uint32_t);
  }
  byteOffsetToEdgeData += localEdgeBegin * sizeof(uint32_t);

  return byteOffsetToEdgeData;
}

std::pair<uint64_t, uint64_t> getLocalAssignment(uint64_t numToSplit) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  return galois::block_range((uint64_t)0, numToSplit, hostID, totalNumHosts);
}
