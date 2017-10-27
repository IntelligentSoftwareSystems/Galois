/** Distributed graph converter helpers (header) -*- C++ -*-
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
 * Distributed graph converter helper signatures/definitions.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#ifndef _GALOIS_DIST_CONVERT_HELP_
#define _GALOIS_DIST_CONVERT_HELP_

#include <fstream>
#include <mpi.h>

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/runtime/Network.h"

/**
 * Wrapper for MPI calls that return an error code. Make sure it is success
 * else die.
 *
 * @param errcode error code returned by an mpi call
 */
void MPICheck(int errcode);


/**
 * TODO documentation
 */
std::vector<std::pair<uint64_t, uint64_t>> 
  getHostToNodeMapping(const uint64_t numHosts, const uint64_t totalNumNodes);

/**
 * Get the assigned host of some node given its global id.
 *
 * @param gID global ID of a node
 * @param hostToNodes Vector containing information about which host has which
 * nodes
 * @returns Host that requested node resides on or -1 if it couldn't be found
 */
uint32_t findHostID(const uint64_t gID, 
                  const std::vector<std::pair<uint64_t, uint64_t>> hostToNodes);

/**
 * Returns the file size of an ifstream.
 *
 * @param openFile an open ifstream
 * @returns file size in bytes of the ifstream
 */
uint64_t getFileSize(std::ifstream& openFile);

/**
 * TODO
 */
std::pair<uint64_t, uint64_t> determineByteRange(std::ifstream& edgeListFile,
                                                 uint64_t fileSize);

/**
 * Determine/send to each host how many edges they should expect to receive
 * from the caller (i.e. this host).
 *
 * TODO params
 */
void sendEdgeCounts(
  const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
  uint64_t localNumEdges, const std::vector<uint32_t>& localEdges
); 

/**
 * Receive the messages from other hosts that tell this host how many edges
 * it should expect to receive. Should be called after sendEdgesCounts.
 *
 * TODO params
 */
uint64_t receiveEdgeCounts();

/**
 * TODO
 */
void sendAssignedEdges( 
  const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
  uint64_t localNumEdges, const std::vector<uint32_t>& localEdges,
  std::vector<std::vector<uint32_t>>& localSrcToDest,
  std::vector<std::mutex>& nodeLocks
);

/**
 * TODO
 */
void receiveAssignedEdges(std::atomic<uint64_t>& edgesToReceive,
    const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
    std::vector<std::vector<uint32_t>>& localSrcToDest,
    std::vector<std::mutex>& nodeLocks);

/**
 * Send/receive other hosts number of assigned edges.
 *
 * TODO
 */
std::vector<uint64_t> getEdgesPerHost(uint64_t localAssignedEdges);

#endif
