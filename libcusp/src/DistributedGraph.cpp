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

/**
 * @file DistributedGraph.cpp
 * Contains command line argument defines for the distributed runtime.
 */

#include <galois/graphs/DistributedGraph.h>

namespace cll = llvm::cl;

//! Command line definition for masters_distribution
cll::opt<MASTERS_DISTRIBUTION> masters_distribution(
    "balanceMasters", cll::desc("Type of masters distribution."),
    cll::values(clEnumValN(BALANCED_MASTERS, "nodes", "Balance nodes only"),
                clEnumValN(BALANCED_EDGES_OF_MASTERS, "edges",
                           "Balance edges only (default)"),
                clEnumValN(BALANCED_MASTERS_AND_EDGES, "both",
                           "Balance both nodes and edges"),
                clEnumValEnd),
    cll::init(BALANCED_EDGES_OF_MASTERS), cll::Hidden);

//! Command line definition for nodeWeightOfMaster
cll::opt<uint32_t>
    nodeWeightOfMaster("nodeWeight",
                       cll::desc("Determines weight of nodes when "
                                 "distributing masterst to hosts"),
                       cll::init(0), cll::Hidden);

//! Command line definition for edgeWeightOfMaster
cll::opt<uint32_t>
    edgeWeightOfMaster("edgeWeight",
                       cll::desc("Determines weight of edges when "
                                 "distributing masters to hosts"),
                       cll::init(0), cll::Hidden);

//! Command line definition for nodeAlphaRanges
cll::opt<uint32_t> nodeAlphaRanges("nodeAlphaRanges",
                                   cll::desc("Determines weight of nodes when "
                                             "partitioning among threads"),
                                   cll::init(0), cll::Hidden);

//! Command line definition for numFileThreads
cll::opt<unsigned>
    numFileThreads("ft",
                   cll::desc("Number of file reading threads or I/O "
                             "requests per host"),
                   cll::init(4), cll::Hidden);

//! Command line definition for edgePartitionSendBufSize
cll::opt<unsigned>
    edgePartitionSendBufSize("edgeBufferSize",
                             cll::desc("Buffer size for batching edges to "
                                       "send during partitioning."),
                             cll::init(8388608), cll::Hidden);

//! Number of rounds to split master assignment phase in in CuSP
//! @todo move this to CuSP source and not here
cll::opt<uint32_t> stateRounds("stateRounds",
                               cll::desc("Frequency of updates in "
                                         "partitioning"),
                               cll::init(1), cll::Hidden);

//! If true, CuSP will use asynchronous synchronization for master assignment
//! phase (phase0)
//! @todo move this to CuSP source and not here
cll::opt<bool> cuspAsync("asyncMasterAssignment",
                         cll::desc("Use async synchronization for CuSP "
                                   "master assignment (default true)"),
                         cll::init(true), cll::Hidden);

//! If true, CuSP activates a barrier before the edge inspection sends.
//! @todo move this to CuSP source and not here
cll::opt<bool> inspectionBarrier("inspectionBarrier",
                                 cll::desc("Activates barrier before CuSP edge "
                                           "inspection (default false)"),
                                 cll::init(false), cll::Hidden);
