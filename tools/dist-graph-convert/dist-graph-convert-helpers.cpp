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
findUniqueSourceNodes(const std::vector<uint32_t>& localEdges) {
  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;

  printf("[%lu] Finding unique nodes\n", hostID);
  galois::substrate::PerThreadStorage<std::set<uint64_t>> threadUniqueNodes;

  uint64_t localNumEdges = localEdges.size() / 2;
  galois::do_all(
    galois::iterate((uint64_t)0, localNumEdges),
    [&] (uint64_t edgeIndex) {
      std::set<uint64_t>& localSet = *threadUniqueNodes.getLocal();
      // src node
      localSet.insert(localEdges[edgeIndex * 2]);
    },
    galois::loopname("FindUniqueNodes"),
    galois::no_stats(),
    galois::steal<false>(),
    galois::timeit()
  );

  std::set<uint64_t> uniqueNodes;

  for (unsigned i = 0; i < threadUniqueNodes.size(); i++) {
    auto& tSet = *threadUniqueNodes.getRemote(i);
    for (auto nodeID : tSet) {
      uniqueNodes.insert(nodeID);
    }
  }

  printf("[%lu] Unique nodes found\n", hostID);

  return uniqueNodes;
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

void accumulateLocalEdgesToChunks(const std::set<uint64_t>& uniqueChunks,
             const std::vector<uint32_t>& localEdges,
             const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode,
             std::vector<uint64_t>& chunkCounts) {
  std::map<uint64_t, galois::GAccumulator<uint64_t>> chunkToAccumulator;
  for (auto chunkID : uniqueChunks) {
    // default-initialize necessary GAccumulators
    chunkToAccumulator[chunkID];
  }

  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;
  printf("[%lu] Chunk accumulators created\n", hostID);

  uint64_t localNumEdges = localEdges.size() / 2;
  // determine which chunk edges go to
  galois::do_all(
    galois::iterate((uint64_t)0, localNumEdges),
    [&] (uint64_t edgeIndex) {
      uint32_t src = localEdges[edgeIndex * 2];
      uint32_t chunkNum = findOwner(src, chunkToNode);
      GALOIS_ASSERT(chunkNum != (uint32_t)-1);
      chunkToAccumulator[chunkNum] += 1;
    },
    galois::loopname("ChunkInspection"),
    galois::no_stats(),
    galois::steal<false>(),
    galois::timeit()
  );

  printf("[%lu] Chunk accumulators done accumulating\n", hostID);

  // update chunk count
  for (auto chunkID : uniqueChunks) {
    chunkCounts[chunkID] = chunkToAccumulator[chunkID].reduce();
  }
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

std::vector<uint64_t> getChunkEdgeCounts(uint64_t numNodeChunks,
                const std::set<uint64_t>& uniqueChunks,
                const std::vector<uint32_t>& localEdges,
                const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode) {
  std::vector<uint64_t> chunkCounts;
  chunkCounts.assign(numNodeChunks, 0);

  accumulateLocalEdgesToChunks(uniqueChunks, localEdges, chunkToNode, 
                               chunkCounts);
  sendAndReceiveEdgeChunkCounts(chunkCounts);

  return chunkCounts;
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
      printf("Host %lu gets nodes %lu to %lu\n", h, lowerNode, upperNode);
    }

    finalMapping.emplace_back(std::pair<uint64_t, uint64_t>(lowerNode, 
                                                            upperNode));
  }

  return finalMapping;
}


std::vector<std::pair<uint64_t, uint64_t>> getEvenNodeToHostMapping(
    const std::vector<uint32_t>& localEdges, uint64_t totalNodeCount, 
    uint64_t totalEdgeCount
) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  uint64_t numNodeChunks = totalEdgeCount / totalNumHosts;
  // TODO better heuristics: basically we don't want to run out of memory,
  // so keep number of chunks from growing too large
  while (numNodeChunks > 10000000) {
    numNodeChunks /= 2;
  }

  std::vector<std::pair<uint64_t, uint64_t>> chunkToNode;

  if (hostID == 0) {
    printf("Num chunks is %lu\n", numNodeChunks);
  }

  for (unsigned i = 0; i < numNodeChunks; i++) {
    chunkToNode.emplace_back(
      galois::block_range((uint64_t)0, (uint64_t)totalNodeCount, i, 
                          numNodeChunks)
    );
  }

  printf("[%lu] Determining edge to chunk counts\n", hostID);
  std::set<uint64_t> uniqueNodes = findUniqueSourceNodes(localEdges);
  std::set<uint64_t> uniqueChunks = findUniqueChunks(uniqueNodes, chunkToNode);
  std::vector<uint64_t> chunkCounts = 
       getChunkEdgeCounts(numNodeChunks, uniqueChunks, localEdges, chunkToNode);
  printf("[%lu] Edge to chunk counts determined\n", hostID);

  // prefix sum on the chunks (reuse array to save memory)
  for (unsigned i = 1; i < numNodeChunks; i++) {
    chunkCounts[i] += chunkCounts[i - 1];
  }

  // to make access to chunkToNode's last element correct with regard to later
  // access (without this access to chunkToNode[chunkSize] is out of bounds)
  chunkToNode.emplace_back(std::pair<uint64_t, uint64_t>(totalNodeCount, 
                                                         totalNodeCount));

  std::vector<std::pair<uint64_t, uint64_t>> finalMapping = 
      getChunkToHostMapping(chunkCounts, chunkToNode);

  return finalMapping;
}

void sendEdgeCounts(
    const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
    const std::vector<uint32_t>& localEdges
) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Determinining edge counts\n", hostID);

  std::vector<galois::GAccumulator<uint64_t>> numEdgesPerHost(totalNumHosts);

  uint64_t localNumEdges = localEdges.size() / 2;
  // determine to which host each edge will go
  galois::do_all(
    galois::iterate((uint64_t)0, localNumEdges),
    [&] (uint64_t edgeIndex) {
      uint32_t src = localEdges[edgeIndex * 2];
      uint32_t edgeOwner = findOwner(src, hostToNodes);
      numEdgesPerHost[edgeOwner] += 1;
    },
    galois::loopname("EdgeInspection"),
    galois::no_stats(),
    galois::steal<false>(),
    galois::timeit()
  );

  printf("[%lu] Sending edge counts\n", hostID);

  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID) continue;
    galois::runtime::SendBuffer b;
    galois::runtime::gSerialize(b, numEdgesPerHost[h].reduce());
    net.sendTagged(h, galois::runtime::evilPhase, b);
  }
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

// TODO make implementation smaller/cleaner i.e. refactor
void sendAssignedEdges(
    const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
    const std::vector<uint32_t>& localEdges,
    std::vector<std::vector<uint32_t>>& localSrcToDest,
    std::vector<std::mutex>& nodeLocks)
{
  auto& net = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID = net.ID;
  uint64_t totalNumHosts = net.Num;

  printf("[%lu] Going to send assigned edges\n", hostID);

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

  printf("[%lu] Passing through edges and assigning\n", hostID);

  uint64_t localNumEdges = localEdges.size() / 2;
  // determine to which host each edge will go
  galois::do_all(
    galois::iterate((uint64_t)0, localNumEdges),
    [&] (uint64_t edgeIndex) {
      uint32_t src = localEdges[edgeIndex * 2];
      uint32_t edgeOwner = findOwner(src, hostToNodes);
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

  printf("[%lu] Buffer cleanup\n", hostID);

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

  printf("[%lu] Going to receive assigned edges\n", hostID);

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
            GALOIS_ASSERT(findOwner(src, hostToNodes) == hostID);
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
                        std::vector<uint64_t>& edgePrefixSum) {
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

void writeEdgeDestData(MPI_File& gr, uint64_t localNumNodes, 
                       uint64_t edgeDestOffset,
                       std::vector<std::vector<uint32_t>>& localSrcToDest) {
  MPI_Status writeStatus;
  for (unsigned i = 0; i < localNumNodes; i++) {
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
