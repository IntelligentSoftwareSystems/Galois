/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
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
 * @file GluonSubstrate.h
 *
 * Contains the implementation for GluonSubstrate.
 */

#ifndef _GALOIS_GLUONSUB_H_
#define _GALOIS_GLUONSUB_H_

#include <unordered_map>
#include <fstream>

#include "galois/runtime/GlobalObj.h"
#include "galois/runtime/DistStats.h"
#include "galois/runtime/SyncStructures.h"
#include "galois/runtime/DataCommMode.h"
#include "galois/DynamicBitset.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/cuda/HostDecls.h"
#endif

#include "galois/runtime/BareMPI.h"
#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
//! bare_mpi type to use; see options in runtime/BareMPI.h
BareMPI bare_mpi = BareMPI::noBareMPI;
#endif

extern DataCommMode enforcedDataMode;

//! Enumeration for specifiying write location for sync calls
enum WriteLocation {
  //! write at source
  writeSource,
  //! write at destination
  writeDestination,
  //! write at source and/or destination
  writeAny
};
//! Enumeration for specifiying read location for sync calls
enum ReadLocation {
  //! read at source
  readSource,
  //! read at destination
  readDestination,
  //! read at source and/or destination
  readAny
};

namespace galois {
namespace graphs {

/**
 * Gluon communication substrate that handles communication given a user graph.
 * User graph should provide certain things the substrate expects.
 *
 * TODO documentation on expected things
 *
 * @tparam GraphTy User graph to handle communication for
 */
template <typename GraphTy>
class GluonSubstrate : public galois::runtime::GlobalObject {
private:
  //! Synchronization type
  enum SyncType {
    syncReduce,   //!< Reduction sync
    syncBroadcast //!< Broadcast sync
  };

  //! Graph name used for printing things
  constexpr static const char* const RNAME = "Gluon";

  //! The graph to handle communication for
  GraphTy& userGraph;
  const unsigned id; //!< Copy of net.ID, which is the ID of the machine.
  bool transposed;  //!< Marks if passed in graph is transposed or not.
  bool isVertexCut;  //!< Marks if passed in graph's partitioning is vertex cut.
  std::pair<unsigned, unsigned> cartesianGrid;  //!< cartesian grid (if any)
  bool partitionAgnostic; //!< true if communication should ignore partitioning
  DataCommMode substrateDataMode; //!< datamode to enforce
  const uint32_t numHosts; //!< Copy of net.Num, which is the total number of machines
  uint32_t num_run;   //!< Keep track of number of runs.
  uint32_t num_round; //!< Keep track of number of rounds.
  bool isCartCut; //!< True of graph is a cartesian cut

  // bitvector status hasn't been maintained
  //! Typedef used so galois::runtime::BITVECTOR_STATUS doesn't have to be
  //! written
  using BITVECTOR_STATUS = galois::runtime::BITVECTOR_STATUS;
  //! A pointer set during syncOnDemand calls that points to the status
  //! of a bitvector with regard to where data has been synchronized
  //! @todo pass the flag as function paramater instead
  BITVECTOR_STATUS* currentBVFlag;

  // memoization optimization
  //! Master nodes on different hosts. For broadcast;
  std::vector<std::vector<size_t>> masterNodes;
  //! Mirror nodes on different hosts. For reduce; comes from the user graph
  //! during initialization (we expect user to give to us)
  std::vector<std::vector<size_t>>& mirrorNodes;
  //! Maximum size of master or mirror nodes on different hosts
  size_t maxSharedSize;

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  std::vector<MPI_Group> mpi_identity_groups;
#endif
  // Used for efficient comms
  galois::DynamicBitSet syncBitset;
  galois::PODResizeableArray<unsigned int> syncOffsets;

  /**
   * Reset a provided bitset given the type of synchronization performed
   *
   * @param syncType Type of synchronization to consider when doing reset
   * @param bitset_reset_range Function to reset range with
   */
  void reset_bitset(SyncType syncType,
                    void (*bitset_reset_range)(size_t, size_t)) {
    size_t numMasters = userGraph.numMasters();
    if (numMasters > 0) {
      // note this assumes masters are from 0 -> a number; CuSP should
      // do this automatically
      if (syncType == syncBroadcast) { // reset masters
        bitset_reset_range(0, numMasters - 1);
      } else {
        assert(syncType == syncReduce);
        // mirrors occur after masters
        if (numMasters < userGraph.size()) {
          bitset_reset_range(numMasters, userGraph.size() - 1);
        }
      }
    } else { // all things are mirrors
      // only need to reset if reduce
      if (syncType == syncReduce) {
        if (userGraph.size() > 0) {
          bitset_reset_range(0, userGraph.size() - 1);
        }
      }
    }
  }

  //! Increments evilPhase, a phase counter used by communication.
  void inline incrementEvilPhase() {
    ++galois::runtime::evilPhase;
    // limit defined by MPI or LCI
    if (galois::runtime::evilPhase >= static_cast<uint32_t>(std::numeric_limits<int16_t>::max())) {
      galois::runtime::evilPhase = 1;
    }
  }

////////////////////////////////////////////////////////////////////////////////
// Proxy communication setup
////////////////////////////////////////////////////////////////////////////////
  /**
   * Let other hosts know about which host has what mirrors/masters;
   * used for later communication of mirrors/masters.
   */
  void exchangeProxyInfo() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    // send off the mirror nodes
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id) continue;

      galois::runtime::SendBuffer b;
      gSerialize(b, mirrorNodes[x]);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    // receive the mirror nodes
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id)
        continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      galois::runtime::gDeserialize(p->second, masterNodes[p->first]);
    }
    incrementEvilPhase();
  }

  /**
   * Send statistics about master/mirror nodes to each host, and
   * report the statistics.
   */
  void sendInfoToHost() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    uint64_t global_total_mirror_nodes = userGraph.size() - userGraph.numMasters();
    uint64_t global_total_owned_nodes  = userGraph.numMasters();

    // send info to host
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id)
        continue;

      galois::runtime::SendBuffer b;
      gSerialize(b, global_total_mirror_nodes, global_total_owned_nodes);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    // receive
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id)
        continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      uint64_t total_mirror_nodes_from_others;
      uint64_t total_owned_nodes_from_others;
      galois::runtime::gDeserialize(p->second, total_mirror_nodes_from_others,
                                    total_owned_nodes_from_others);
      global_total_mirror_nodes += total_mirror_nodes_from_others;
      global_total_owned_nodes += total_owned_nodes_from_others;
    }
    incrementEvilPhase();

    assert(userGraph.globalSize() == global_total_owned_nodes);
    // report stats
    if (net.ID == 0) {
      reportProxyStats(global_total_mirror_nodes, global_total_owned_nodes);
    }
  }

  /**
   * Sets up the communication between the different hosts that contain
   * different parts of the graph by exchanging master/mirror information.
   */
  void setupCommunication() {
    galois::CondStatTimer<MORE_DIST_STATS> Tcomm_setup("CommunicationSetupTime",
                                                       RNAME);

    // barrier so that all hosts start the timer together
    galois::runtime::getHostBarrier().wait();

    Tcomm_setup.start();

    // Exchange information for memoization optimization.
    exchangeProxyInfo();
    // convert the global ids stored in the master/mirror nodes arrays to local
    // ids
    // TODO: use 32-bit distinct vectors for masters and mirrors from here on
    for (uint32_t h = 0; h < masterNodes.size(); ++h) {
      galois::do_all(
          galois::iterate(size_t{0}, masterNodes[h].size()),
          [&](size_t n) {
            masterNodes[h][n] = userGraph.getLID(masterNodes[h][n]);
          },
#if MORE_COMM_STATS
          galois::loopname(get_run_identifier("MasterNodes").c_str()),
#endif
          galois::no_stats());
    }

    for (uint32_t h = 0; h < mirrorNodes.size(); ++h) {
      galois::do_all(
          galois::iterate(size_t{0}, mirrorNodes[h].size()),
          [&](size_t n) {
            mirrorNodes[h][n] = userGraph.getLID(mirrorNodes[h][n]);
          },
#if MORE_COMM_STATS
          galois::loopname(get_run_identifier("MirrorNodes").c_str()),
#endif
          galois::no_stats());
    }

    Tcomm_setup.stop();

    maxSharedSize = 0;
    // report masters/mirrors to/from other hosts as statistics
    for (auto x = 0U; x < masterNodes.size(); ++x) {
      if (x == id) continue;
      std::string master_nodes_str =
          "MasterNodesFrom_" + std::to_string(id) + "_To_" + std::to_string(x);
      galois::runtime::reportStatCond_Tsum<MORE_DIST_STATS>(
          RNAME, master_nodes_str, masterNodes[x].size());
      if (masterNodes[x].size() > maxSharedSize) {
        maxSharedSize = masterNodes[x].size();
      }
    }

    for (auto x = 0U; x < mirrorNodes.size(); ++x) {
      if (x == id) continue;
      std::string mirror_nodes_str =
          "MirrorNodesFrom_" + std::to_string(x) + "_To_" + std::to_string(id);
      galois::runtime::reportStatCond_Tsum<MORE_DIST_STATS>(
          RNAME, mirror_nodes_str, mirrorNodes[x].size());
      if (mirrorNodes[x].size() > maxSharedSize) {
        maxSharedSize = mirrorNodes[x].size();
      }
    }

    sendInfoToHost();

    // do not track memory usage of partitioning
    auto& net = galois::runtime::getSystemNetworkInterface();
    net.resetMemUsage();
  }

  /**
   * Reports master/mirror stats.
   * Assumes that communication has already occured so that the host
   * calling it actually has the info required.
   *
   * @param global_total_mirror_nodes number of mirror nodes on all hosts
   * @param global_total_owned_nodes number of "owned" nodes on all hosts
   */
  void reportProxyStats(uint64_t global_total_mirror_nodes,
                        uint64_t global_total_owned_nodes) {
    float replication_factor =
        (float)(global_total_mirror_nodes + userGraph.globalSize()) /
        (float)userGraph.globalSize();
    galois::runtime::reportStat_Single(RNAME, "ReplicationFactor",
                                       replication_factor);

    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(
        RNAME, "TotalNodes", userGraph.globalSize());
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(
        RNAME, "TotalGlobalMirrorNodes", global_total_mirror_nodes);
  }

////////////////////////////////////////////////////////////////////////////////
// Initializers
////////////////////////////////////////////////////////////////////////////////
  /**
   * Initalize MPI related things. The MPI layer itself should have been
   * initialized when the network interface was initiailized.
   */
  void initBareMPI() {
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    if (bare_mpi == noBareMPI)
      return;

#ifdef GALOIS_USE_LWCI
    // sanity check of ranks
    int taskRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);
    if ((unsigned)taskRank != id) GALOIS_DIE("Mismatch in MPI rank");
    int numTasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    if ((unsigned)numTasks != numHosts) GALOIS_DIE("Mismatch in MPI rank");
#endif
    // group setup
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    mpi_identity_groups.resize(numHosts);

    for (unsigned x = 0; x < numHosts; ++x) {
      const int g[1] = {(int)x};
      MPI_Group_incl(world_group, 1, g, &mpi_identity_groups[x]);
    }

    if (id == 0) {
      switch (bare_mpi) {
      case nonBlockingBareMPI:
        galois::gPrint("Using non-blocking bare MPI\n");
        break;
      case oneSidedBareMPI:
        galois::gPrint("Using one-sided bare MPI\n");
        break;
      case noBareMPI:
      default:
        GALOIS_DIE("Unsupported bare MPI");
      }
    }
#endif
  }

public:
  /**
   * Delete default constructor: this class NEEDS to have a graph passed into
   * it.
   */
  GluonSubstrate() = delete;

  /**
   * Constructor for GluonSubstrate. Initializes metadata fields.
   *
   * @param host host number that this graph resides on
   * @param numHosts total number of hosts in the currently executing program
   * @param _transposed True if the graph is transposed
   */
  GluonSubstrate(GraphTy& _userGraph, unsigned host, unsigned numHosts,
            bool _transposed,
            std::pair<unsigned, unsigned> _cartesianGrid=std::make_pair(0u, 0u),
            bool _partitionAgnostic=false,
            DataCommMode _enforcedDataMode=DataCommMode::noData)
      : galois::runtime::GlobalObject(this),
        userGraph(_userGraph),
        id(host),
        transposed(_transposed),
        isVertexCut(userGraph.is_vertex_cut()),
        cartesianGrid(_cartesianGrid),
        partitionAgnostic(_partitionAgnostic),
        substrateDataMode(_enforcedDataMode),
        numHosts(numHosts),
        num_run(0),
        num_round(0),
        currentBVFlag(nullptr),
        mirrorNodes(userGraph.getMirrorNodes()) {
    if (cartesianGrid.first != 0 && cartesianGrid.second != 0) {
      GALOIS_ASSERT(cartesianGrid.first * cartesianGrid.second == numHosts,
                    "Cartesian split doesn't equal number of hosts");
      if (id == 0) {
        galois::gInfo("Gluon optimizing communication for 2-D cartesian cut: ",
                      cartesianGrid.first, " x ", cartesianGrid.second);
      }
      isCartCut = true;
    } else {
      assert(cartesianGrid.first == 0 && cartesianGrid.second == 0);
      isCartCut = false;
    }

    // set this global value for use on GPUs mostly
    // TODO find a better way to do this without globals
    enforcedDataMode = _enforcedDataMode;

    initBareMPI();
    // master setup from mirrors done by setupCommunication call
    masterNodes.resize(numHosts);
    // setup proxy communication
    galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct_comm(
        "GraphCommSetupTime", RNAME);
    Tgraph_construct_comm.start();
    setupCommunication();
    Tgraph_construct_comm.stop();
  }

////////////////////////////////////////////////////////////////////////////////
// Data extraction from bitsets
////////////////////////////////////////////////////////////////////////////////

private:
  /**
   * Given a bitset, determine the indices of the bitset that are currently
   * set.
   *
   * @tparam syncType either reduce or broadcast; only used to name the timer
   *
   * @param loopName string used to name the timer for this function
   * @param bitset_comm the bitset to get the offsets of
   * @param offsets output: the offset vector that will contain indices into
   * the bitset that are set
   * @param bit_set_count output: will be set to the number of bits set in the
   * bitset
   */
  template <SyncType syncType>
  void getOffsetsFromBitset(const std::string& loopName,
                            const galois::DynamicBitSet& bitset_comm,
                            galois::PODResizeableArray<unsigned int>& offsets,
                            size_t& bit_set_count) const {
    // timer creation
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string offsets_timer_str(syncTypeStr + "Offsets_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Toffsets(offsets_timer_str.c_str(),
                                                    RNAME);

    Toffsets.start();

    auto activeThreads = galois::getActiveThreads();
    std::vector<unsigned int> t_prefix_bit_counts(activeThreads);

    // count how many bits are set on each thread
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      // TODO use block_range instead
      unsigned int block_size = bitset_comm.size() / nthreads;
      if ((bitset_comm.size() % nthreads) > 0) ++block_size;
      assert((block_size * nthreads) >= bitset_comm.size());

      unsigned int start = tid * block_size;
      unsigned int end   = (tid + 1) * block_size;
      if (end > bitset_comm.size())
        end = bitset_comm.size();

      unsigned int count = 0;
      for (unsigned int i = start; i < end; ++i) {
        if (bitset_comm.test(i))
          ++count;
      }

      t_prefix_bit_counts[tid] = count;
    });

    // calculate prefix sum of bits per thread
    for (unsigned int i = 1; i < activeThreads; ++i) {
      t_prefix_bit_counts[i] += t_prefix_bit_counts[i - 1];
    }
    // total num of set bits
    bit_set_count = t_prefix_bit_counts[activeThreads - 1];

    // calculate the indices of the set bits and save them to the offset
    // vector
    if (bit_set_count > 0) {
      offsets.resize(bit_set_count);
      galois::on_each([&](unsigned tid, unsigned nthreads) {
        // TODO use block_range instead
        // TODO this is same calculation as above; maybe refactor it
        // into function?
        unsigned int block_size = bitset_comm.size() / nthreads;
        if ((bitset_comm.size() % nthreads) > 0)
          ++block_size;
        assert((block_size * nthreads) >= bitset_comm.size());

        unsigned int start = tid * block_size;
        unsigned int end   = (tid + 1) * block_size;
        if (end > bitset_comm.size())
          end = bitset_comm.size();

        unsigned int count = 0;
        unsigned int t_prefix_bit_count;
        if (tid == 0) {
          t_prefix_bit_count = 0;
        } else {
          t_prefix_bit_count = t_prefix_bit_counts[tid - 1];
        }

        for (unsigned int i = start; i < end; ++i) {
          if (bitset_comm.test(i)) {
            offsets[t_prefix_bit_count + count] = i;
            ++count;
          }
        }
      });
    }
    Toffsets.stop();
  }

  /**
   * Determine what data needs to be synchronized based on the passed in
   * bitset_compute and returns information regarding these need-to-be-sync'd
   * nodes.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done;
   * only used to get the size of the type being synchronized in this function
   * @tparam syncType type of synchronization this function is being called
   * for; only used to name a timer
   *
   * @param loopName loopname used to name the timer for the function
   * @param indices A vector that contains the local ids of the nodes that
   * you want to potentially synchronize
   * @param bitset_compute Contains the full bitset of all nodes in this
   * graph
   * @param bitset_comm OUTPUT: bitset that marks which indices in the passed
   * in indices array need to be synchronized
   * @param offsets OUTPUT: contains indices into bitset_comm that are set
   * @param bit_set_count OUTPUT: contains number of bits set in bitset_comm
   * @param data_mode OUTPUT: the way that this data should be communicated
   * based on how much data needs to be sent out
   */
  template <typename FnTy, SyncType syncType>
  void getBitsetAndOffsets(const std::string& loopName,
                           const std::vector<size_t>& indices,
                           const galois::DynamicBitSet& bitset_compute,
                           galois::DynamicBitSet& bitset_comm,
                           galois::PODResizeableArray<unsigned int>& offsets,
                           size_t& bit_set_count,
                           DataCommMode& data_mode) const {
    if (substrateDataMode != onlyData) {
      bitset_comm.reset();
      std::string syncTypeStr =
          (syncType == syncReduce) ? "Reduce" : "Broadcast";
      std::string doall_str(syncTypeStr + "Bitset_" + loopName);

      bitset_comm.reset();
      // determine which local nodes in the indices array need to be
      // sychronized
      galois::do_all(galois::iterate(size_t{0}, indices.size()),
                     [&](size_t n) {
                       // assumes each lid is unique as test is not thread safe
                       size_t lid = indices[n];
                       if (bitset_compute.test(lid)) {
                         bitset_comm.set(n);
                       }
                     },
#if MORE_COMM_STATS
                     galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                     galois::no_stats());

      // get the number of set bits and the offsets into the comm bitset
      getOffsetsFromBitset<syncType>(loopName, bitset_comm, offsets,
                                     bit_set_count);
    }

    data_mode = get_data_mode<typename FnTy::ValTy>(bit_set_count,
                                                    indices.size());
  }

////////////////////////////////////////////////////////////////////////////////
// Local to global ID conversion
////////////////////////////////////////////////////////////////////////////////
  /**
   * Converts LIDs of nodes we are interested in into GIDs.
   *
   * @tparam syncType either reduce or broadcast; only used to name the timer
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of nodes that we are interested in
   * @param offsets INPUT/OUTPUT holds offsets into "indices" that we should
   * use; after function completion, holds global ids of nodes we are interested
   * in
   */
  template <SyncType syncType>
  void convertLIDToGID(const std::string& loopName,
                       const std::vector<size_t>& indices,
                       galois::PODResizeableArray<unsigned int>& offsets) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string doall_str(syncTypeStr + "_LID2GID_" +
                          get_run_identifier(loopName));
    galois::do_all(galois::iterate(size_t{0}, offsets.size()),
                   [&](size_t n) {
                     offsets[n] =
                         static_cast<uint32_t>(
                           userGraph.getGID(indices[offsets[n]])
                         );
                   },
#if MORE_COMM_STATS
                   galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                   galois::no_stats());
  }

  /**
   * Converts a vector of GIDs into local ids.
   *
   * @tparam syncType either reduce or broadcast; only used to name the timer
   *
   * @param loopName name of loop used to name timer
   * @param offsets holds GIDs to convert to LIDs
   */
  template <SyncType syncType>
  void convertGIDToLID(const std::string& loopName,
                       galois::PODResizeableArray<unsigned int>& offsets) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string doall_str(syncTypeStr + "_GID2LID_" +
                          get_run_identifier(loopName));

    galois::do_all(galois::iterate(size_t{0}, offsets.size()),
                   [&](size_t n) {
                     offsets[n] = userGraph.getLID(offsets[n]);
                   },
#if MORE_COMM_STATS
                   galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                   galois::no_stats());
  }

////////////////////////////////////////////////////////////////////////////////
// Message prep functions (buffering, send buffer getting, etc.)
////////////////////////////////////////////////////////////////////////////////
  /**
   * Get data that is going to be sent for synchronization and returns
   * it in a send buffer.
   *
   * @tparam syncType synchronization type
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has information needed to access bitset
   *
   * @param loopName Name to give timer
   * @param x Host to send to
   * @param b OUTPUT: Buffer that will hold data to send
   */
  template <
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy,
      typename VecTy, bool async,
      typename std::enable_if<!BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  void getSendBuffer(std::string loopName, unsigned x,
                     galois::runtime::SendBuffer& b) {
    auto& sharedNodes = (syncType == syncReduce) ? mirrorNodes : masterNodes;

    if (BitsetFnTy::is_valid()) {
      syncExtract<syncType, SyncFnTy, BitsetFnTy, VecTy, async>(loopName, x,
                                                         sharedNodes[x], b);
    } else {
      syncExtract<syncType, SyncFnTy, VecTy, async>(loopName, x, sharedNodes[x], b);
    }

    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string statSendBytes_str(syncTypeStr + "SendBytes_" +
                                  get_run_identifier(loopName));

    galois::runtime::reportStat_Tsum(RNAME, statSendBytes_str, b.size());
  }
  template <
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
      typename VecTy, bool async,
      typename std::enable_if<BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  void getSendBuffer(std::string loopName, unsigned x,
                     galois::runtime::SendBuffer& b) {
    auto& sharedNodes = (syncType == syncReduce) ? mirrorNodes : masterNodes;

    syncExtract<syncType, SyncFnTy, BitsetFnTy, VecTy, async>(loopName, x,
                                                       sharedNodes[x], b);

    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string statSendBytes_str(syncTypeStr + "SendBytesVector_" +
                                  get_run_identifier(loopName));

    galois::runtime::reportStat_Tsum(RNAME, statSendBytes_str, b.size());
  }

  /**
   * Given data to serialize in val_vec, serialize it into the send buffer
   * depending on the mode of data communication selected for the data.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam VecType type of val_vec, which stores the data to send
   *
   * @param loopName loop name used for timers
   * @param data_mode the way that the data should be communicated
   * @param bit_set_count the number of items we are sending in this message
   * @param indices list of all nodes that we are potentially interested in
   * sending things to
   * @param offsets contains indicies into "indices" that we are interested in
   * @param val_vec contains the data that we are serializing to send
   * @param b the buffer in which to serialize the message we are sending
   * to
   */
  template <bool async, SyncType syncType, typename VecType>
  void serializeMessage(std::string loopName, DataCommMode data_mode,
                        size_t bit_set_count, std::vector<size_t>& indices,
                        galois::PODResizeableArray<unsigned int>& offsets,
                        galois::DynamicBitSet& bit_set_comm, VecType& val_vec,
                        galois::runtime::SendBuffer& b) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string serialize_timer_str(syncTypeStr + "SerializeMessage_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Tserialize(serialize_timer_str.c_str(),
                                                    RNAME);
    if (data_mode == noData) {
      if (!async) {
        Tserialize.start();
        gSerialize(b, data_mode);
        Tserialize.stop();
      }
    } else if (data_mode == gidsData) {
      offsets.resize(bit_set_count);
      convertLIDToGID<syncType>(loopName, indices, offsets);
      val_vec.resize(bit_set_count);
      Tserialize.start();
      gSerialize(b, data_mode, bit_set_count, offsets, val_vec);
      Tserialize.stop();
    } else if (data_mode == offsetsData) {
      offsets.resize(bit_set_count);
      val_vec.resize(bit_set_count);
      Tserialize.start();
      gSerialize(b, data_mode, bit_set_count, offsets, val_vec);
      Tserialize.stop();
    } else if (data_mode == bitsetData) {
      val_vec.resize(bit_set_count);
      Tserialize.start();
      gSerialize(b, data_mode, bit_set_count, bit_set_comm, val_vec);
      Tserialize.stop();
    } else { // onlyData
      Tserialize.start();
      gSerialize(b, data_mode, val_vec);
      Tserialize.stop();
    }
  }

  /**
   * Given the data mode, deserialize the rest of a message in a Receive Buffer.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam VecType type of val_vec, which data will be deserialized into
   *
   * @param loopName used to name timers for statistics
   * @param data_mode data mode with which the original message was sent;
   * determines how to deserialize the rest of the message
   * @param buf buffer which contains the received message to deserialize
   *
   * The rest of the arguments are output arguments (they are passed by
   * reference)
   *
   * @param bit_set_count Var that holds number of bits set (i.e. number of
   * node changed) after deserialization
   * @param offsets holds offsets data after deserialization if data mode is
   * offsets + data
   * @param bit_set_comm holds the bitset representing changed nodes after
   * deserialization of data mode is bitset + data
   * @param buf_start
   * @param retval
   * @param val_vec The data proper will be deserialized into this vector
   */
  template <SyncType syncType, typename VecType>
  void deserializeMessage(std::string loopName, DataCommMode data_mode,
                       uint32_t num, galois::runtime::RecvBuffer& buf,
                       size_t& bit_set_count,
                       galois::PODResizeableArray<unsigned int>& offsets,
                       galois::DynamicBitSet& bit_set_comm, size_t& buf_start,
                       size_t& retval, VecType& val_vec) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string serialize_timer_str(syncTypeStr + "DeserializeMessage_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Tdeserialize(serialize_timer_str.c_str(),
                                                    RNAME);
    Tdeserialize.start();

    // get other metadata associated with message if mode isn't OnlyData
    if (data_mode != onlyData) {
      galois::runtime::gDeserialize(buf, bit_set_count);

      if (data_mode == gidsData) {
        galois::runtime::gDeserialize(buf, offsets);
        convertGIDToLID<syncType>(loopName, offsets);
      } else if (data_mode == offsetsData) {
        galois::runtime::gDeserialize(buf, offsets);
      } else if (data_mode == bitsetData) {
        bit_set_comm.resize(num);
        galois::runtime::gDeserialize(buf, bit_set_comm);
      } else if (data_mode == dataSplit) {
        galois::runtime::gDeserialize(buf, buf_start);
      } else if (data_mode == dataSplitFirst) {
        galois::runtime::gDeserialize(buf, retval);
      }
    }

    // get data itself
    galois::runtime::gDeserialize(buf, val_vec);

    Tdeserialize.stop();
  }

////////////////////////////////////////////////////////////////////////////////
// Other helper functions
////////////////////////////////////////////////////////////////////////////////

  //! Returns the grid row ID of this host
  unsigned gridRowID() const { return (id / cartesianGrid.second); }
  //! Returns the grid row ID of the specified host
  unsigned gridRowID(unsigned hid) const {
    return (hid / cartesianGrid.second);
  }
  //! Returns the grid column ID of this host
  unsigned gridColumnID() const { return (id % cartesianGrid.second); }
  //! Returns the grid column ID of the specified host
  unsigned gridColumnID(unsigned hid) const {
    return (hid % cartesianGrid.second);
  }

  /**
   * Determine if a host is a communication partner using cartesian grid.
   */
  bool isNotCommPartnerCVC(unsigned host, SyncType syncType,
                           WriteLocation writeLocation,
                           ReadLocation readLocation) {
    assert(cartesianGrid.first != 0);
    assert(cartesianGrid.second != 0);

    if (transposed) {
      if (syncType == syncReduce) {
        switch (writeLocation) {
        case writeSource:
          return (gridColumnID() != gridColumnID(host));
        case writeDestination:
          return (gridRowID() != gridRowID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          GALOIS_DIE("isNotCommPartnerCVC error");
        }
      } else { // syncBroadcast
        switch (readLocation) {
        case readSource:
          return (gridColumnID() != gridColumnID(host));
        case readDestination:
          return (gridRowID() != gridRowID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          GALOIS_DIE("isNotCommPartnerCVC error");
        }
      }
    } else {
      if (syncType == syncReduce) {
        switch (writeLocation) {
        case writeSource:
          return (gridRowID() != gridRowID(host));
        case writeDestination:
          return (gridColumnID() != gridColumnID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          GALOIS_DIE("isNotCommPartnerCVC error");
        }
      } else { // syncBroadcast, 1
        switch (readLocation) {
        case readSource:
          return (gridRowID() != gridRowID(host));
        case readDestination:
          return (gridColumnID() != gridColumnID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          GALOIS_DIE("isNotCommPartnerCVC error");
        }
      }
      return false;
    }
  }

  // Requirement: For all X and Y,
  // On X, nothingToSend(Y) <=> On Y, nothingToRecv(X)
  /**
   * Determine if we have anything that we need to send to a particular host
   *
   * @param host Host number that we may or may not send to
   * @param syncType Synchronization type to determine which nodes on a
   * host need to be considered
   * @param writeLocation If data is being written to on source or
   * destination (or both)
   * @param readLocation If data is being read from on source or
   * destination (or both)
   * @returns true if there is nothing to send to a host, false otherwise
   */
  bool nothingToSend(unsigned host, SyncType syncType,
                     WriteLocation writeLocation, ReadLocation readLocation) {
    auto& sharedNodes = (syncType == syncReduce) ? mirrorNodes : masterNodes;
    // TODO refactor (below)
    if (!isCartCut) {
      return (sharedNodes[host].size() == 0);
    } else {
      // TODO If CVC, call is not comm partner else use default above
      if (sharedNodes[host].size() > 0) {
        return isNotCommPartnerCVC(host, syncType, writeLocation,
                                   readLocation);
      } else {
        return true;
      }
    }
  }

  /**
   * Determine if we have anything that we need to receive from a particular
   * host
   *
   * @param host Host number that we may or may not receive from
   * @param syncType Synchronization type to determine which nodes on a
   * host need to be considered
   * @param writeLocation If data is being written to on source or
   * destination (or both)
   * @param readLocation If data is being read from on source or
   * destination (or both)
   * @returns true if there is nothing to receive from a host, false otherwise
   */
  bool nothingToRecv(unsigned host, SyncType syncType,
                     WriteLocation writeLocation, ReadLocation readLocation) {
    auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
    // TODO refactor (above)
    if (!isCartCut) {
      return (sharedNodes[host].size() == 0);
    } else {
      if (sharedNodes[host].size() > 0) {
        return isNotCommPartnerCVC(host, syncType, writeLocation,
                                   readLocation);
      } else {
        return true;
      }
    }
  }

  /**
   * Reports bytes saved by using the bitset to only selectively load data
   * to send.
   *
   * @tparam SyncFnTy synchronization structure with info needed to synchronize;
   * used for size calculation
   *
   * @param loopName loop name used for timers
   * @param syncTypeStr String used to name timers
   * @param totalToSend Total amount of nodes that are potentially sent (not
   * necessarily all nodees will be sent)
   * @param bitSetCount Number of nodes that will actually be sent
   * @param bitSetComm bitset used to send data
   */
  template <typename SyncFnTy>
  void reportRedundantSize(std::string loopName, std::string syncTypeStr,
                           uint32_t totalToSend, size_t bitSetCount,
                           const galois::DynamicBitSet& bitSetComm) {
    size_t redundant_size =
        (totalToSend - bitSetCount) * sizeof(typename SyncFnTy::ValTy);
    size_t bit_set_size = (bitSetComm.get_vec().size() * sizeof(uint64_t));

    if (redundant_size > bit_set_size) {
      std::string statSavedBytes_str(syncTypeStr + "SavedBytes_" +
                                     get_run_identifier(loopName));

      galois::runtime::reportStatCond_Tsum<MORE_DIST_STATS>(
          RNAME, statSavedBytes_str, (redundant_size - bit_set_size));
    }
  }

////////////////////////////////////////////////////////////////////////////////
// Extract data from nodes (for reduce and broadcast)
////////////////////////////////////////////////////////////////////////////////
  /**
   * Extracts data at provided lid.
   *
   * This version (reduce) resets the value after extract.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam syncType either reduce or broadcast; determines if reset is
   * necessary
   *
   * @param lid local id of node to get data from
   * @returns data (specified by FnTy) of node with local id lid
   */
  /* Reduction extract resets the value afterwards */
  template <typename FnTy, SyncType syncType>
  inline typename FnTy::ValTy extractWrapper(size_t lid) {
    if (syncType == syncReduce) {
      auto val = FnTy::extract(lid, userGraph.getData(lid));
      FnTy::reset(lid, userGraph.getData(lid));
      return val;
    } else {
      return FnTy::extract(lid, userGraph.getData(lid));
    }
  }

  /**
   * Extracts data at provided lid; uses vecIndex to get the correct element
   * from the vector.
   *
   * This version (reduce) resets the value after extract.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam syncType either reduce or broadcast; determines if reset is
   * necessary
   *
   * @param lid local id of node to get data from
   * @param vecIndex index to grab from vector in node
   * @returns data (specified by FnTy) of node with local id lid
   */
  /* Reduction extract resets the value afterwards */
  template <typename FnTy, SyncType syncType>
  inline typename FnTy::ValTy extractWrapper(size_t lid, unsigned vecIndex) {
    if (syncType == syncReduce) {
      auto val = FnTy::extract(lid, userGraph.getData(lid), vecIndex);
      FnTy::reset(lid, userGraph.getData(lid), vecIndex);
      return val;
    } else {
      return FnTy::extract(lid, userGraph.getData(lid), vecIndex);
    }
  }

  /**
   * Based on provided arguments, extracts the data that we are interested
   * in sending into val_vec.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam syncType either reduce or broadcast; used to determine if reseting
   * the extracted field is necessary
   * @tparam identity_offsets If this is true, then ignore the offsets
   * array and just grab directly from indices (i.e. don't pick out
   * particular elements, just grab contiguous chunk)
   * @tparam parallelize Determines if parallelizing the extraction is done or
   * not
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of nodes that we are interested in
   * @param size Number of elements to extract
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param val_vec OUTPUT: holds the extracted data
   * @param start Offset into val_vec to start saving data to
   */
  template <typename FnTy, SyncType syncType, typename VecTy, bool identity_offsets = false,
            bool parallelize = true>
  void extractSubset(const std::string& loopName,
                     const std::vector<size_t>& indices, size_t size,
                     const galois::PODResizeableArray<unsigned int>& offsets,
                     VecTy& val_vec,
                     size_t start = 0) {
    if (parallelize) {
      std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
      std::string doall_str(syncTypeStr + "ExtractVal_" + loopName);

      galois::do_all(galois::iterate(start, start + size),
                     [&](unsigned int n) {
                       unsigned int offset;
                       if (identity_offsets) offset = n;
                       else offset = offsets[n];
                       size_t lid = indices[offset];
                       val_vec[n - start] = extractWrapper<FnTy, syncType>(lid);
                     },
#if MORE_COMM_STATS
                     galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                     galois::no_stats());
    } else { // non-parallel version
      for (unsigned n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets) offset = n;
        else offset = offsets[n];

        size_t lid         = indices[offset];
        val_vec[n - start] = extractWrapper<FnTy, syncType>(lid);
      }
    }
  }

  /**
   * Based on provided arguments, extracts the data that we are interested
   * in sending into val_vec. Same as above, except it has the vecIndex
   * arguments and requires vecSync to be true
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam syncType either reduce or broadcast; used to determine if reseting
   * the extracted field is necessary
   * @tparam identity_offsets If this is true, then ignore the offsets
   * array and just grab directly from indices (i.e. don't pick out
   * particular elements, just grab contiguous chunk)
   * @tparam parallelize Determines if parallelizing the extraction is done or
   * not
   * @tparam vecSync Only set to true if the field being synchronized is a
   * vector and synchronization is occuring element by element. MUST BE SET
   * TO TRUE IN ORDER FOR THIS FUNCTION TO COMPILE.
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of nodes that we are interested in
   * @param size Number of elements to extract
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param val_vec OUTPUT: holds the extracted data
   * @param vecIndex which element of the vector to extract from node
   * @param start Offset into val_vec to start saving data to
   */
  // TODO find a better way to have this variant without code duplication
  template <typename FnTy, SyncType syncType, typename VecTy, bool identity_offsets = false,
            bool parallelize = true, bool vecSync = false,
            typename std::enable_if<vecSync>::type* = nullptr>
  void extractSubset(const std::string& loopName,
                     const std::vector<size_t>& indices, size_t size,
                     const galois::PODResizeableArray<unsigned int>& offsets,
                     VecTy& val_vec,
                     unsigned vecIndex, size_t start = 0) {
    val_vec.resize(size); // resize val vec for this vecIndex

    if (parallelize) {
      std::string syncTypeStr =
          (syncType == syncReduce) ? "Reduce" : "Broadcast";
      std::string doall_str(syncTypeStr + "ExtractValVector_" + loopName);

      galois::do_all(galois::iterate(start, start + size),
                     [&](unsigned int n) {
                       unsigned int offset;
                       if (identity_offsets) offset = n;
                       else offset = offsets[n];
                       size_t lid = indices[offset];
                       val_vec[n - start] =
                           extractWrapper<FnTy, syncType>(lid, vecIndex);
                     },
#if MORE_COMM_STATS
                     galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                     galois::no_stats());
    } else { // non-parallel version
      for (unsigned n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets) offset = n;
        else offset = offsets[n];
        size_t lid         = indices[offset];
        val_vec[n - start] = extractWrapper<FnTy, syncType>(lid, vecIndex);
      }
    }
  }

  /**
   * Based on provided arguments, extracts the data that we are interested
   * in sending into a send buffer. Lazy serialize variant that works with
   * certain SeqTy.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SeqTy Type of sequence that we are getting data from
   * @tparam syncType either reduce or broadcast; used to determine if reseting
   * the extracted field is necessary
   * @tparam identity_offsets If this is true, then ignore the offsets
   * array and just grab directly from indices (i.e. don't pick out
   * particular elements, just grab contiguous chunk)
   * @tparam parallelize Determines if parallelizing the extraction is done or
   * not
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of nodes that we are interested in
   * @param size Number of elements to extract
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param b send buffer to extract data into
   * @param lseq sequence to get data from
   * @param start Offset into send buffer to start saving data to
   */
  template <typename FnTy, typename SeqTy, SyncType syncType,
            bool identity_offsets = false, bool parallelize = true>
  void extractSubset(const std::string& loopName,
                     const std::vector<size_t>& indices, size_t size,
                     const galois::PODResizeableArray<unsigned int>& offsets,
                     galois::runtime::SendBuffer& b, SeqTy lseq,
                     size_t start = 0) {
    if (parallelize) {
      std::string syncTypeStr =
          (syncType == syncReduce) ? "Reduce" : "Broadcast";
      std::string doall_str(syncTypeStr + "ExtractVal_" + loopName);

      galois::do_all(galois::iterate(start, start + size),
                     [&](unsigned int n) {
                       unsigned int offset;
                       if (identity_offsets) offset = n;
                       else offset = offsets[n];

                       size_t lid = indices[offset];
                       gSerializeLazy(b, lseq, n - start,
                                      extractWrapper<FnTy, syncType>(lid));
                     },
#if MORE_COMM_STATS
                     galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                     galois::no_stats());
    } else {
      for (unsigned int n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets) offset = n;
        else offset = offsets[n];
        size_t lid = indices[offset];
        gSerializeLazy(b, lseq, n - start,
                       extractWrapper<FnTy, syncType>(lid));
      }
    }
  }

  /**
   * GPU wrap function: extracts data from nodes and resets them to the
   * reduction identity value as specified by the sync structure. (Reduce only)
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SyncType Must be reduce
   *
   * @param x node id to extract from
   * @param v vector to extract data to
   *
   * @returns true if called on GPU device
   */
  template <typename FnTy, SyncType syncType>
  inline bool extractBatchWrapper(unsigned x, galois::runtime::SendBuffer& b) {
    if (syncType == syncReduce) {
      return FnTy::extract_reset_batch(x, b.getVec().data());
    } else {
      return FnTy::extract_batch(x, b.getVec().data());
    }
  }

  /**
   * GPU wrap function: extracts data from nodes and resets them to the
   * reduction identity value as specified by the sync structure. (Reduce only)
   *
   * This version specifies more arguments.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SyncType Must be reduce
   *
   * @param x node id to extract from
   * @param b
   * @param o
   * @param v
   * @param s
   * @param data_mode
   *
   * @returns true if called on GPU device
   */
  template <typename FnTy, SyncType syncType>
  inline bool extractBatchWrapper(unsigned x, galois::runtime::SendBuffer& b,
                                  size_t& s, DataCommMode& data_mode) {
    if (syncType == syncReduce) {
      return FnTy::extract_reset_batch(x, b.getVec().data(), &s, &data_mode);
    } else {
      return FnTy::extract_batch(x, b.getVec().data(), &s, &data_mode);
    }
  }

////////////////////////////////////////////////////////////////////////////////
// Reduce/sets on node (for broadcast)
////////////////////////////////////////////////////////////////////////////////
  /**
   * Reduce variant. Takes a value and reduces it according to the sync
   * structure provided to the function.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam syncType Reduce sync or broadcast sync
   *
   * @param lid local id of node to reduce to
   * @param val value to reduce to
   * @param bit_set_compute bitset indicating which nodes have changed; updated
   * if reduction causes a change
   */
  template <typename FnTy, SyncType syncType, bool async>
  inline void setWrapper(size_t lid, typename FnTy::ValTy val,
                         galois::DynamicBitSet& bit_set_compute) {
    if (syncType == syncReduce) {
      if (FnTy::reduce(lid, userGraph.getData(lid), val)) {
        if (bit_set_compute.size() != 0) bit_set_compute.set(lid);
      }
    } else {
      if (async) FnTy::reduce(lid, userGraph.getData(lid), val);
      else FnTy::setVal(lid, userGraph.getData(lid), val);
    }
  }

  /**
   * VECTOR VARIANT.
   *
   * Reduce variant. Takes a value and reduces it according to the sync
   * structure provided to the function. Only reduces the element at a
   * particular index of the vector field being sychronized.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam syncType Reduce sync or broadcast sync
   *
   * @param lid local id of node to reduce to
   * @param val value to reduce to
   * @param bit_set_compute bitset indicating which nodes have changed; updated
   * if reduction causes a change
   * @param vecIndex which element of the vector to reduce in the node
   */
  template <typename FnTy, SyncType syncType, bool async>
  inline void setWrapper(size_t lid, typename FnTy::ValTy val,
                         galois::DynamicBitSet& bit_set_compute,
                         unsigned vecIndex) {
    if (syncType == syncReduce) {
      if (FnTy::reduce(lid, userGraph.getData(lid), val, vecIndex)) {
        if (bit_set_compute.size() != 0)
          bit_set_compute.set(lid);
      }
    } else {
      if (async) FnTy::reduce(lid, userGraph.getData(lid), val, vecIndex);
      else FnTy::setVal(lid, userGraph.getData(lid), val, vecIndex);
    }
  }

  /**
   * Given data received from another host and information on which nodes
   * to update, do the reduce/set of the received data to update local nodes.
   *
   * Complement function, in some sense, of extractSubset.
   *
   * @tparam VecTy type of indices variable
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SyncType Reduce or broadcast
   * @tparam identity_offsets If this is true, then ignore the offsets
   * array and just grab directly from indices (i.e. don't pick out
   * particular elements, just grab contiguous chunk)
   * @tparam parallelize True if updates to nodes are to be parallelized
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of nodes that we are interested in
   * @param size Number of elements to set
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param val_vec holds data we will use to set
   * @param bit_set_compute bitset indicating which nodes have changed
   * @param start Offset into val_vec to start saving data to
   */
  template <typename IndicesVecTy, typename FnTy, SyncType syncType, 
            typename VecTy, bool async,
            bool identity_offsets = false, bool parallelize = true>
  void setSubset(const std::string& loopName, const IndicesVecTy& indices,
                 size_t size, const galois::PODResizeableArray<unsigned int>& offsets,
                 VecTy& val_vec,
                 galois::DynamicBitSet& bit_set_compute, size_t start = 0) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string doall_str(syncTypeStr + "SetVal_" +
                          get_run_identifier(loopName));

    if (parallelize) {
      galois::do_all(galois::iterate(start, start + size),
                     [&](unsigned int n) {
                       unsigned int offset;
                       if (identity_offsets) offset = n;
                       else offset = offsets[n];
                       auto lid = indices[offset];
                       setWrapper<FnTy, syncType, async>(lid, val_vec[n - start],
                                                         bit_set_compute);
                     },
#if MORE_COMM_STATS
                     galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                     galois::no_stats());
    } else {
      for (unsigned int n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets) offset = n;
        else offset = offsets[n];
        auto lid = indices[offset];
        setWrapper<FnTy, syncType, async>(lid, val_vec[n - start],
                                          bit_set_compute);
      }
    }
  }

  /**
   * VECTOR BITSET VARIANT.
   *
   * Given data received from another host and information on which nodes
   * to update, do the reduce/set of the received data to update local nodes.
   * It will only update a single index of the vector specified by the
   * sync structures at a time.
   *
   * Complement function, in some sense, of extractSubset, vector bitset
   * variant.
   *
   * @tparam VecTy type of indices variable
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SyncType Reduce or broadcast
   * @tparam identity_offsets If this is true, then ignore the offsets
   * array and just grab directly from indices (i.e. don't pick out
   * particular elements, just grab contiguous chunk)
   * @tparam parallelize True if updates to nodes are to be parallelized
   * @tparam vecSync Only set to true if the field being synchronized is a
   * vector. MUST BE SET TO TRUE FOR THIS FUNCTION TO COMPILE
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of nodes that we are interested in
   * @param size Number of elements to set
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param val_vec holds data we will use to set
   * @param bit_set_compute bitset indicating which nodes have changed
   * @param vecIndex which element of the vector to set in the node
   * @param start Offset into val_vec to start saving data to
   */
  // TODO find a better way to have this variant without code duplication
  template <typename IndicesVecTy, typename FnTy, SyncType syncType, 
            typename VecTy, bool async,
            bool identity_offsets = false, bool parallelize = true,
            bool vecSync                            = false,
            typename std::enable_if<vecSync>::type* = nullptr>
  void setSubset(const std::string& loopName, const IndicesVecTy& indices,
                 size_t size, const galois::PODResizeableArray<unsigned int>& offsets,
                 VecTy& val_vec,
                 galois::DynamicBitSet& bit_set_compute, unsigned vecIndex,
                 size_t start = 0) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string doall_str(syncTypeStr + "SetValVector_" +
                          get_run_identifier(loopName));

    if (parallelize) {
      galois::do_all(galois::iterate(start, start + size),
                     [&](unsigned int n) {
                       unsigned int offset;
                       if (identity_offsets) offset = n;
                       else offset = offsets[n];
                       auto lid = indices[offset];
                       setWrapper<FnTy, syncType, async>(lid,
                             val_vec[n - start], bit_set_compute, vecIndex);
                     },
#if MORE_COMM_STATS
                     galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
                     galois::no_stats());
    } else {
      for (unsigned int n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets) offset = n;
        else offset = offsets[n];
        auto lid = indices[offset];
        setWrapper<FnTy, syncType, async>(lid, val_vec[n - start],
                                          bit_set_compute, vecIndex);
      }
    }
  }

  /**
   * GPU wrapper function to reduce multiple nodes at once.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SyncType Must be reduce
   *
   * @param x node id to set
   * @param v
   *
   * @returns true if called on GPU device
   */
  template <typename FnTy, SyncType syncType, bool async>
  inline bool setBatchWrapper(unsigned x, galois::runtime::RecvBuffer& b) {
    if (syncType == syncReduce) {
      return FnTy::reduce_batch(x, b.getVec().data() + b.getOffset());
    } else {
      if (async) {
        return FnTy::reduce_mirror_batch(x, b.getVec().data() + b.getOffset());
      } else {
        return FnTy::setVal_batch(x, b.getVec().data() + b.getOffset());
      }
    }
  }

  /**
   * GPU wrapper function to reduce multiple nodes at once. More detailed
   * arguments.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SyncType Must be reduce
   *
   * @param x node id to set
   * @param b
   * @param o
   * @param v
   * @param s
   * @param data_mode
   *
   * @returns true if called on GPU device
   */
  template <typename FnTy, SyncType syncType, bool async>
  inline bool setBatchWrapper(unsigned x, galois::runtime::RecvBuffer& b,
                              DataCommMode& data_mode) {
    if (syncType == syncReduce) {
      return FnTy::reduce_batch(x, b.getVec().data() + b.getOffset(), data_mode);
    } else {
      if (async) {
        return FnTy::reduce_mirror_batch(x, b.getVec().data() + b.getOffset(),
                                         data_mode);
      } else {
        return FnTy::setVal_batch(x, b.getVec().data() + b.getOffset(),
                                  data_mode);
      }
    }
  }

////////////////////////////////////////////////////////////////////////////////
// Sends
////////////////////////////////////////////////////////////////////////////////
  /**
   * Non-bitset extract that uses serializelazy to copy data over to the
   * buffer. REQUIRES that the ValTy be memory copyable.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam syncFnTy struct that has info on how to do synchronization
   *
   * @param loopName loop name used for timers
   * @param from_id
   * @param indices Vector that contains node ids of nodes that we will
   * potentially send things to
   * @param b OUTPUT: buffer that will be sent over the network; contains data
   * based on set bits in bitset
   */
  template <SyncType syncType, typename SyncFnTy, 
            typename VecTy, bool async,
            typename std::enable_if<galois::runtime::is_memory_copyable<
                typename SyncFnTy::ValTy>::value>::type* = nullptr>
  void syncExtract(std::string loopName, unsigned from_id,
                   std::vector<size_t>& indices,
                   galois::runtime::SendBuffer& b) {
    uint32_t num = indices.size();
    static VecTy val_vec; // sometimes wasteful
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string extract_timer_str(syncTypeStr + "Extract_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textract(extract_timer_str.c_str(),
                                                    RNAME);
    std::string extract_batch_timer_str(syncTypeStr + "ExtractBatch_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textractbatch(
        extract_batch_timer_str.c_str(), RNAME);

    DataCommMode data_mode;

    Textract.start();

    if (num > 0) {
      data_mode = onlyData;
      b.reserve(sizeof(DataCommMode)
          + sizeof(size_t)
          + (num * sizeof(typename SyncFnTy::ValTy)));

      Textractbatch.start();
      bool batch_succeeded =
          extractBatchWrapper<SyncFnTy, syncType>(from_id, b);
      Textractbatch.stop();

      if (!batch_succeeded) {
        b.resize(0);
        val_vec.reserve(maxSharedSize);
        val_vec.resize(num);
        gSerialize(b, onlyData);
        auto lseq = gSerializeLazySeq(
            b, num, (galois::PODResizeableArray<typename SyncFnTy::ValTy>*)nullptr);
        extractSubset<SyncFnTy, decltype(lseq), syncType, true, true>(
            loopName, indices, num, offsets, b, lseq);
      } else {
        b.resize(sizeof(DataCommMode)
            + sizeof(size_t)
            + (num * sizeof(typename SyncFnTy::ValTy)));
      }
    } else {
      data_mode = noData;
      b.resize(0);
      if (!async) {
        gSerialize(b, noData);
      }
    }

    Textract.stop();

    std::string metadata_str(syncTypeStr + "MetadataMode_" +
                             std::to_string(data_mode) + "_" +
                             get_run_identifier(loopName));
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(RNAME,
                                                            metadata_str, 1);
  }

  /**
   * Non-bitset extract for when the type of the item being sync'd isn't
   * memory copyable.
   *
   * Extracts all of the data for all nodes in indices and saves it into
   * a send buffer for return.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam syncFnTy struct that has info on how to do synchronization
   *
   * @param loopName loop name used for timers
   * @param from_id
   * @param indices Vector that contains node ids of nodes that we will
   * potentially send things to
   * @param b OUTPUT: buffer that will be sent over the network; contains data
   * based on set bits in bitset
   */
  template <SyncType syncType, typename SyncFnTy, 
            typename VecTy, bool async,
            typename std::enable_if<!galois::runtime::is_memory_copyable<
                typename SyncFnTy::ValTy>::value>::type* = nullptr>
  void syncExtract(std::string loopName, unsigned from_id,
                   std::vector<size_t>& indices,
                   galois::runtime::SendBuffer& b) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string extract_timer_str(syncTypeStr + "Extract_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textract(extract_timer_str.c_str(),
                                                    RNAME);
    std::string extract_batch_timer_str(syncTypeStr + "ExtractBatch_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textractbatch(
        extract_batch_timer_str.c_str(), RNAME);

    DataCommMode data_mode;

    uint32_t num = indices.size();
    static VecTy val_vec; // sometimes wasteful
    static galois::PODResizeableArray<unsigned int> dummyVector;

    Textract.start();

    if (num > 0) {
      data_mode = onlyData;
      b.reserve(sizeof(DataCommMode)
          + sizeof(size_t)
          + (num * sizeof(typename SyncFnTy::ValTy)));

      Textractbatch.start();
      bool batch_succeeded =
          extractBatchWrapper<SyncFnTy, syncType>(from_id, b);
      Textractbatch.stop();

      if (!batch_succeeded) {
        b.resize(0);
        val_vec.reserve(maxSharedSize);
        val_vec.resize(num);
        // get everything (note I pass in "indices" as offsets as it won't
        // even get used anyways)
        extractSubset<SyncFnTy, syncType, VecTy, true, true>(loopName, indices, num,
                                                       dummyVector, val_vec);
        gSerialize(b, onlyData, val_vec);
      } else {
        b.resize(sizeof(DataCommMode)
            + sizeof(size_t)
            + (num * sizeof(typename SyncFnTy::ValTy)));
      }

    } else {
      b.resize(0);
      if (!async) {
        data_mode = noData;
        gSerialize(b, noData);
      }
    }

    Textract.stop();

    std::string metadata_str(syncTypeStr + "MetadataMode_" +
                             std::to_string(data_mode) + "_" +
                             get_run_identifier(loopName));
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(RNAME,
                                                            metadata_str, 1);
  }

  /**
   * Extracts the data that will be sent to a host in this round of
   * synchronization based on the passed in bitset and saves it to a
   * send buffer.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam syncFnTy struct that has info on how to do synchronization
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   * being used for the extraction
   *
   * @param loopName loop name used for timers
   * @param from_id
   * @param indices Vector that contains node ids of nodes that we will
   * potentially send things to
   * @param b OUTPUT: buffer that will be sent over the network; contains data
   * based on set bits in bitset
   */
  template <
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
      typename VecTy, bool async,
      typename std::enable_if<!BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  void syncExtract(std::string loopName, unsigned from_id,
                   std::vector<size_t>& indices,
                   galois::runtime::SendBuffer& b) {
    uint32_t num = indices.size();
    galois::DynamicBitSet& bit_set_comm = syncBitset;
    static VecTy val_vec; // sometimes wasteful
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;

    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string extract_timer_str(syncTypeStr + "Extract_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textract(extract_timer_str.c_str(),
                                                    RNAME);
    std::string extract_alloc_timer_str(syncTypeStr + "ExtractAlloc_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textractalloc(
        extract_alloc_timer_str.c_str(), RNAME);
    std::string extract_batch_timer_str(syncTypeStr + "ExtractBatch_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textractbatch(
        extract_batch_timer_str.c_str(), RNAME);

    DataCommMode data_mode;

    Textract.start();

    if (num > 0) {
      size_t bit_set_count = 0;
      Textractalloc.start();
      if (substrateDataMode == gidsData) {
        b.reserve(sizeof(DataCommMode)
            + sizeof(bit_set_count)
            + sizeof(size_t)
            + (num * sizeof(unsigned int))
            + sizeof(size_t)
            + (num * sizeof(typename SyncFnTy::ValTy)));
      } else if (substrateDataMode == offsetsData) {
        b.reserve(sizeof(DataCommMode)
            + sizeof(bit_set_count)
            + sizeof(size_t)
            + (num * sizeof(unsigned int))
            + sizeof(size_t)
            + (num * sizeof(typename SyncFnTy::ValTy)));
      } else if (substrateDataMode == bitsetData) {
        size_t bitset_alloc_size =
            ((num + 63) / 64) * sizeof(uint64_t);
        b.reserve(sizeof(DataCommMode)
            + sizeof(bit_set_count)
            + sizeof(size_t) // bitset size
            + sizeof(size_t) // bitset vector size
            + bitset_alloc_size
            + sizeof(size_t)
            + (num * sizeof(typename SyncFnTy::ValTy)));
      } else { // onlyData or noData (auto)
        size_t bitset_alloc_size =
            ((num + 63) / 64) * sizeof(uint64_t);
        b.reserve(sizeof(DataCommMode)
            + sizeof(bit_set_count)
            + sizeof(size_t) // bitset size
            + sizeof(size_t) // bitset vector size
            + bitset_alloc_size
            + sizeof(size_t)
            + (num * sizeof(typename SyncFnTy::ValTy)));
      }
      Textractalloc.stop();

      Textractbatch.start();
      bool batch_succeeded = extractBatchWrapper<SyncFnTy, syncType>(
          from_id, b, bit_set_count, data_mode);
      Textractbatch.stop();

      // GPUs have a batch function they can use; CPUs do not; therefore,
      // CPUS always enter this if block
      if (!batch_succeeded) {
        Textractalloc.start();
        b.resize(0);
        bit_set_comm.reserve(maxSharedSize);
        offsets.reserve(maxSharedSize);
        val_vec.reserve(maxSharedSize);
        bit_set_comm.resize(num);
        offsets.resize(num);
        val_vec.resize(num);
        Textractalloc.stop();
        const galois::DynamicBitSet& bit_set_compute = BitsetFnTy::get();

        getBitsetAndOffsets<SyncFnTy, syncType>(
            loopName, indices, bit_set_compute, bit_set_comm, offsets,
            bit_set_count, data_mode);

        if (data_mode == onlyData) {
          bit_set_count = indices.size();
          extractSubset<SyncFnTy, syncType, VecTy, true, true>(
              loopName, indices, bit_set_count, offsets, val_vec);
        } else if (data_mode !=
                   noData) { // bitsetData or offsetsData or gidsData
          extractSubset<SyncFnTy, syncType, VecTy, false, true>(
              loopName, indices, bit_set_count, offsets, val_vec);
        }
        serializeMessage<async, syncType>(loopName, data_mode, bit_set_count, indices,
                                   offsets, bit_set_comm, val_vec, b);
      } else {
        if (data_mode == noData) {
          b.resize(0);
          if (!async) {
            gSerialize(b, data_mode);
          }
        } else if (data_mode == gidsData) {
          b.resize(sizeof(DataCommMode)
              + sizeof(bit_set_count)
              + sizeof(size_t)
              + (bit_set_count * sizeof(unsigned int))
              + sizeof(size_t)
              + (bit_set_count * sizeof(typename SyncFnTy::ValTy)));
        } else if (data_mode == offsetsData) {
          b.resize(sizeof(DataCommMode)
              + sizeof(bit_set_count)
              + sizeof(size_t)
              + (bit_set_count * sizeof(unsigned int))
              + sizeof(size_t)
              + (bit_set_count * sizeof(typename SyncFnTy::ValTy)));
        } else if (data_mode == bitsetData) {
          size_t bitset_alloc_size =
              ((num + 63) / 64) * sizeof(uint64_t);
          b.resize(sizeof(DataCommMode)
              + sizeof(bit_set_count)
              + sizeof(size_t) // bitset size
              + sizeof(size_t) // bitset vector size
              + bitset_alloc_size
              + sizeof(size_t)
              + (bit_set_count * sizeof(typename SyncFnTy::ValTy)));
        } else { // onlyData
          b.resize(sizeof(DataCommMode)
              + sizeof(size_t)
              + (num * sizeof(typename SyncFnTy::ValTy)));
        }
      }

      reportRedundantSize<SyncFnTy>(loopName, syncTypeStr, num, bit_set_count,
                                    bit_set_comm);
    } else {
      data_mode = noData;
      b.resize(0);
      if (!async) {
        gSerialize(b, noData);
      }
    }

    Textract.stop();

    std::string metadata_str(syncTypeStr + "MetadataMode_" +
                             std::to_string(data_mode) + "_" +
                             get_run_identifier(loopName));
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(RNAME,
                                                            metadata_str, 1);
  }

  /**
   * Vector bitset variant.
   *
   * Extracts the data that will be sent to a host in this round of
   * synchronization based on the passed in bitset and saves it to a
   * send buffer. Unlike other variants, this will extract an entire
   * vector element by element.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam syncFnTy struct that has info on how to do synchronization
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   * being used for the extraction. MUST BE A VECTOR BITSET
   *
   * @param loopName loop name used for timers
   * @param from_id
   * @param indices Vector that contains node ids of nodes that we will
   * potentially send things to
   * @param b OUTPUT: buffer that will be sent over the network; contains data
   * based on set bits in bitset
   */
  template <
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
      typename VecTy, bool async,
      typename std::enable_if<BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  void syncExtract(std::string loopName, unsigned from_id,
                   std::vector<size_t>& indices,
                   galois::runtime::SendBuffer& b) {
    uint32_t num = indices.size();
    galois::DynamicBitSet& bit_set_comm = syncBitset;
    static VecTy val_vec; // sometimes wasteful
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;

    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string extract_timer_str(syncTypeStr + "ExtractVector_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Textract(extract_timer_str.c_str(),
                                                    RNAME);

    Textract.start();

    if (num > 0) {
      bit_set_comm.reserve(maxSharedSize);
      offsets.reserve(maxSharedSize);
      val_vec.reserve(maxSharedSize);
      bit_set_comm.resize(num);
      offsets.resize(num);
      val_vec.resize(num);
    }

    DataCommMode data_mode;
    // loop over all bitsets in the vector of bitsets; each one corresponds to
    // a different index in the vector field we are synchronizing
    for (unsigned i = 0; i < BitsetFnTy::numBitsets(); i++) {
      if (num > 0) {
        bit_set_comm.reset();

        size_t bit_set_count = 0;

        // No GPU support currently
        const galois::DynamicBitSet& bit_set_compute = BitsetFnTy::get(i);

        getBitsetAndOffsets<SyncFnTy, syncType>(
            loopName, indices, bit_set_compute, bit_set_comm, offsets,
            bit_set_count, data_mode);

        // note the extra template argument which specifies that this is a
        // vector extract, i.e. get element i of the vector (i passed in as
        // argument as well)
        if (data_mode == onlyData) {
          // galois::gInfo(id, " node ", i, " has data to send");
          bit_set_count = indices.size();
          extractSubset<SyncFnTy, syncType, VecTy, true, true, true>(
              loopName, indices, bit_set_count, offsets, val_vec, i);
        } else if (data_mode !=
                   noData) { // bitsetData or offsetsData or gidsData
          // galois::gInfo(id, " node ", i, " has data to send");
          extractSubset<SyncFnTy, syncType, VecTy, false, true, true>(
              loopName, indices, bit_set_count, offsets, val_vec, i);
        }

        reportRedundantSize<SyncFnTy>(loopName, syncTypeStr, num, bit_set_count,
                                      bit_set_comm);
        serializeMessage<async, syncType>(loopName, data_mode, bit_set_count, indices,
                                   offsets, bit_set_comm, val_vec, b);
      } else {
        if (!async) { // TODO: is this fine?
          // append noData for however many bitsets there are
          gSerialize(b, noData);
        }
      }
    }

    Textract.stop();

    // FIXME report metadata mode for the different bitsets?
    // std::string metadata_str(syncTypeStr + "_METADATA_MODE" +
    //                         std::to_string(data_mode) +
    //                         get_run_identifier(loopName));
    // galois::runtime::reportStat_Single(RNAME, metadata_str, 1);
  }


#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  /**
   * Sync using MPI instead of network layer.
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_send(std::string loopName) {
    static std::vector<galois::runtime::SendBuffer> b;
    static std::vector<MPI_Request> request;
    b.resize(numHosts);
    request.resize(numHosts, MPI_REQUEST_NULL);

    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType, writeLocation, readLocation))
        continue;

      int ready = 0;
      MPI_Test(&request[x], &ready, MPI_STATUS_IGNORE);
      if (!ready) {
        assert(b[x].size() > 0);
        MPI_Wait(&request[x], MPI_STATUS_IGNORE);
      }
      if (b[x].size() > 0) {
        b[x].getVec().clear();
      }

      getSendBuffer<syncType, SyncFnTy, BitsetFnTy>(loopName, x, b[x]);

      MPI_Isend((uint8_t*)b[x].linearData(), b[x].size(), MPI_BYTE, x, 32767,
                MPI_COMM_WORLD, &request[x]);
    }

    if (BitsetFnTy::is_valid()) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }
  }

  /**
   * Sync put using MPI instead of network layer
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_put(std::string loopName, const MPI_Group& mpi_access_group,
                    const std::vector<MPI_Win>& window) {

    MPI_Win_start(mpi_access_group, 0, window[id]);

    std::vector<galois::runtime::SendBuffer> b(numHosts);
    std::vector<size_t> size(numHosts);
    uint64_t send_buffers_size = 0;

    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType, writeLocation, readLocation))
        continue;

      getSendBuffer<syncType, SyncFnTy, BitsetFnTy>(loopName, x, b[x]);

      size[x] = b[x].size();
      send_buffers_size += size[x];
      MPI_Put((uint8_t*)&size[x], sizeof(size_t), MPI_BYTE, x, 0,
              sizeof(size_t), MPI_BYTE, window[id]);
      MPI_Put((uint8_t*)b[x].linearData(), size[x], MPI_BYTE, x, sizeof(size_t),
              size[x], MPI_BYTE, window[id]);
    }

    auto& net = galois::runtime::getSystemNetworkInterface();
    net.incrementMemUsage(send_buffers_size);

    MPI_Win_complete(window[id]);
    net.decrementMemUsage(send_buffers_size);

    if (BitsetFnTy::is_valid()) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }
  }
#endif

  /**
   * Sends data to all hosts (if there is anything that needs to be sent
   * to that particular host) and adjusts bitset according to sync type.
   *
   * @tparam writeLocation Location data is written (src or dst)
   * @tparam readLocation Location data is read (src or dst)
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has information needed to access bitset
   *
   * @param loopName used to name timers created by this sync send
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
            typename VecTy, bool async>
  void syncNetSend(std::string loopName) {
    static galois::runtime::SendBuffer b; // although a static variable, allocation not reused
                                          // due to std::move in net.sendTagged()

    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string statNumMessages_str(syncTypeStr + "NumMessages_" +
                                  get_run_identifier(loopName));

    size_t numMessages = 0;
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType, writeLocation, readLocation))
        continue;

      getSendBuffer<syncType, SyncFnTy, BitsetFnTy, VecTy, async>(loopName, x, b);

      if ((!async) || (b.size() > 0)) {
        size_t syncTypePhase = 0;
        if (async && (syncType == syncBroadcast)) syncTypePhase = 1;
        net.sendTagged(x, galois::runtime::evilPhase, b, syncTypePhase);
        ++numMessages;
      }
    }
    if (!async) {
      // Will force all messages to be processed before continuing
      net.flush();
    }

    if (BitsetFnTy::is_valid()) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }

    galois::runtime::reportStat_Tsum(
        RNAME, statNumMessages_str, numMessages);
  }

  /**
   * Sends data over the network to other hosts based on the provided template
   * arguments.
   *
   * @tparam writeLocation Location data is written (src or dst)
   * @tparam readLocation Location data is read (src or dst)
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
            typename VecTy, bool async>
  void syncSend(std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<MORE_COMM_STATS> TSendTime(
        (syncTypeStr + "Send_" + get_run_identifier(loopName)).c_str(), RNAME);

    TSendTime.start();
    syncNetSend<writeLocation, readLocation, syncType, SyncFnTy, BitsetFnTy, VecTy, async>(
        loopName);
    TSendTime.stop();
  }

////////////////////////////////////////////////////////////////////////////////
// Receives
////////////////////////////////////////////////////////////////////////////////

  /**
   * Deserializes messages from other hosts and applies them to update local
   * data based on the provided sync structures.
   *
   * Complement of syncExtract.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param from_id ID of host which the message we are processing was received
   * from
   * @param buf Buffer that contains received message from other host
   * @param loopName used to name timers for statistics
   */
  template <
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
      typename VecTy, bool async,
      typename std::enable_if<!BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  size_t syncRecvApply(uint32_t from_id, galois::runtime::RecvBuffer& buf,
                       std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string set_timer_str(syncTypeStr + "Set_" +
                              get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Tset(set_timer_str.c_str(), RNAME);
    std::string set_batch_timer_str(syncTypeStr + "SetBatch_" +
                                    get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Tsetbatch(
        set_batch_timer_str.c_str(), RNAME);

    galois::DynamicBitSet& bit_set_comm = syncBitset;
    static VecTy val_vec;
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;

    auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
    uint32_t num      = sharedNodes[from_id].size();
    size_t retval     = 0;

    Tset.start();

    if (num > 0) { // only enter if we expect message from that host
      DataCommMode data_mode;
      // 1st deserialize gets data mode
      galois::runtime::gDeserialize(buf, data_mode);

      if (data_mode != noData) {
        // GPU update call
        Tsetbatch.start();
        bool batch_succeeded = setBatchWrapper<SyncFnTy, syncType, async>(
            from_id, buf, data_mode);
        Tsetbatch.stop();

        // cpu always enters this block
        if (!batch_succeeded) {
          size_t bit_set_count = num;
          size_t buf_start     = 0;

          // deserialize the rest of the data in the buffer depending on the data
          // mode; arguments passed in here are mostly output vars
          deserializeMessage<syncType>(loopName, data_mode, num, buf, bit_set_count,
                                    offsets, bit_set_comm, buf_start, retval,
                                    val_vec);

          bit_set_comm.reserve(maxSharedSize);
          offsets.reserve(maxSharedSize);
          val_vec.reserve(maxSharedSize);

          galois::DynamicBitSet& bit_set_compute = BitsetFnTy::get();

          if (data_mode == bitsetData) {
            size_t bit_set_count2;
            getOffsetsFromBitset<syncType>(loopName, bit_set_comm, offsets,
                                              bit_set_count2);
            assert(bit_set_count == bit_set_count2);
          }

          if (data_mode == onlyData) {
            setSubset<decltype(sharedNodes[from_id]), SyncFnTy, syncType, VecTy,
                      async, true, true>(
                            loopName, sharedNodes[from_id], bit_set_count,
                            offsets, val_vec, bit_set_compute);
          } else if (data_mode == dataSplit || data_mode == dataSplitFirst) {
            setSubset<decltype(sharedNodes[from_id]), SyncFnTy, syncType, VecTy,
                      async, true, true>(
                            loopName, sharedNodes[from_id], bit_set_count,
                            offsets, val_vec, bit_set_compute, buf_start);
          } else if (data_mode == gidsData) {
            setSubset<decltype(offsets), SyncFnTy, syncType, VecTy,
                      async, true, true>(
                            loopName, offsets, bit_set_count, offsets, val_vec,
                            bit_set_compute);
          } else { // bitsetData or offsetsData
            setSubset<decltype(sharedNodes[from_id]), SyncFnTy, syncType, VecTy,
                      async, false, true>(
                            loopName, sharedNodes[from_id], bit_set_count,
                            offsets, val_vec, bit_set_compute);
          }
          // TODO: reduce could update the bitset, so it needs to be copied
          // back to the device
        }
      }
    }

    Tset.stop();

    return retval;
  }

  /**
   * VECTOR BITSET VARIANT.
   *
   * Deserializes messages from other hosts and applies them to update local
   * data based on the provided sync structures. Each message will contain
   * a series of messages that must be deserialized (the number of such
   * messages corresponds to the size of the vector that is being synchronized).
   *
   * Complement of syncExtract, vector bitset version.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   * MUST BE VECTOR BITSET
   *
   * @param from_id ID of host which the message we are processing was received
   * from
   * @param buf Buffer that contains received message from other host
   * @param loopName used to name timers for statistics
   */
  template <
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
      typename VecTy, bool async,
      typename std::enable_if<BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  size_t syncRecvApply(uint32_t from_id, galois::runtime::RecvBuffer& buf,
                       std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string set_timer_str(syncTypeStr + "SetVector_" +
                              get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Tset(set_timer_str.c_str(), RNAME);

    galois::DynamicBitSet& bit_set_comm = syncBitset;
    static VecTy val_vec;
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;

    auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
    uint32_t num      = sharedNodes[from_id].size();
    size_t retval     = 0;

    Tset.start();

    if (num > 0) { // only enter if we expect message from that host
      for (unsigned i = 0; i < BitsetFnTy::numBitsets(); i++) {
        DataCommMode data_mode;
        // 1st deserialize gets data mode
        galois::runtime::gDeserialize(buf, data_mode);

        if (data_mode != noData) {
          size_t bit_set_count = num;
          size_t buf_start     = 0;

          // deserialize the rest of the data in the buffer depending on the
          // data mode; arguments passed in here are mostly output vars
          deserializeMessage<syncType>(loopName, data_mode, num, buf,
                                    bit_set_count, offsets, bit_set_comm,
                                    buf_start, retval, val_vec);

          galois::DynamicBitSet& bit_set_compute = BitsetFnTy::get(i);

          if (data_mode == bitsetData) {
            size_t bit_set_count2;
            getOffsetsFromBitset<syncType>(loopName, bit_set_comm, offsets,
                                              bit_set_count2);
            assert(bit_set_count == bit_set_count2);
          }

          // Note the extra template argument and i argument which cause
          // execution to deal with a particular element of the vector field
          // we are synchronizing
          if (data_mode == onlyData) {
            setSubset<decltype(sharedNodes[from_id]), SyncFnTy, syncType, VecTy,
                      async, true, true, true>(
                                  loopName, sharedNodes[from_id],
                                  bit_set_count, offsets, val_vec,
                                  bit_set_compute, i);
          } else if (data_mode == dataSplit || data_mode == dataSplitFirst) {
            setSubset<decltype(sharedNodes[from_id]), SyncFnTy, syncType, VecTy, true,
                      async, true, true, true>(
                                  loopName, sharedNodes[from_id],
                                  bit_set_count, offsets, val_vec,
                                  bit_set_compute, i, buf_start);
          } else if (data_mode == gidsData) {
            setSubset<decltype(offsets), SyncFnTy, syncType, VecTy,
                      async, true, true, true>(
                                  loopName, offsets, bit_set_count,
                                  offsets, val_vec,
                                  bit_set_compute, i);
          } else { // bitsetData or offsetsData
            setSubset<decltype(sharedNodes[from_id]), SyncFnTy, syncType, VecTy,
                      async, false, true, true>(
                                  loopName, sharedNodes[from_id],
                                  bit_set_count, offsets, val_vec,
                                  bit_set_compute, i);
          }
        }
      }
    }

    Tset.stop();

    return retval;
  }

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  /**
   * MPI Irecv wrapper for sync
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_recv_post(std::string loopName,
                          std::vector<MPI_Request>& request,
                          const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType, writeLocation, readLocation))
        continue;

      MPI_Irecv((uint8_t*)rb[x].data(), rb[x].size(), MPI_BYTE, x, 32767,
                MPI_COMM_WORLD, &request[x]);
    }
  }

  /**
   * MPI receive wrapper for sync
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_recv_wait(std::string loopName,
                          std::vector<MPI_Request>& request,
                          const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType, writeLocation, readLocation))
        continue;

      MPI_Status status;
      MPI_Wait(&request[x], &status);

      int size = 0;
      MPI_Get_count(&status, MPI_BYTE, &size);

      galois::runtime::RecvBuffer rbuf(rb[x].begin(), rb[x].begin() + size);

      syncRecvApply<syncType, SyncFnTy, BitsetFnTy>(x, rbuf, loopName);
    }
  }

  /**
   * MPI get wrapper for sync
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_get(std::string loopName, const std::vector<MPI_Win>& window,
                    const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType, writeLocation, readLocation))
        continue;

      MPI_Win_wait(window[x]);

      size_t size = 0;
      memcpy(&size, rb[x].data(), sizeof(size_t));

      galois::runtime::RecvBuffer rbuf(rb[x].begin() + sizeof(size_t),
                                       rb[x].begin() + sizeof(size_t) + size);

      MPI_Win_post(mpi_identity_groups[x], 0, window[x]);

      syncRecvApply<syncType, SyncFnTy, BitsetFnTy>(x, rbuf, loopName);
    }
  }
#endif

  /**
   * Determines if there is anything to receive from a host and receives/applies
   * the messages.
   *
   * @tparam writeLocation Location data is written (src or dst)
   * @tparam readLocation Location data is read (src or dst)
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy,
            typename VecTy, bool async>
  void syncNetRecv(std::string loopName) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string wait_timer_str("Wait_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> Twait(wait_timer_str.c_str(),
                                                    RNAME);

    if (async) {
      size_t syncTypePhase = 0;
      if (syncType == syncBroadcast) syncTypePhase = 1;
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase);

        if (p) {
          syncRecvApply<syncType, SyncFnTy, BitsetFnTy, VecTy, async>(p->first,
                                                        p->second,
                                                        loopName);
        }
      } while (p);
    } else {
      for (unsigned x = 0; x < numHosts; ++x) {
        if (x == id)
          continue;
        if (nothingToRecv(x, syncType, writeLocation, readLocation))
          continue;

        Twait.start();
        decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
        do {
          p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        } while (!p);
        Twait.stop();

        syncRecvApply<syncType, SyncFnTy, BitsetFnTy, VecTy, async>(p->first,
                                                      p->second,
                                                      loopName);
      }
      incrementEvilPhase();
    }
  }

  /**
   * Receives messages from all other hosts and "applies" the message (reduce
   * or set) based on the sync structure provided.
   *
   * @tparam writeLocation Location data is written (src or dst)
   * @tparam readLocation Location data is read (src or dst)
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy, 
            typename VecTy, bool async>
  void syncRecv(std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<MORE_COMM_STATS> TRecvTime(
        (syncTypeStr + "Recv_" + get_run_identifier(loopName)).c_str(), RNAME);

    TRecvTime.start();
    syncNetRecv<writeLocation, readLocation, syncType, SyncFnTy, BitsetFnTy, VecTy, async>(
        loopName);
    TRecvTime.stop();
  }

////////////////////////////////////////////////////////////////////////////////
// MPI sync variants
////////////////////////////////////////////////////////////////////////////////
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  /**
   * Nonblocking MPI sync
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void syncNonblockingMPI(std::string loopName,
                            bool use_bitset_to_send = true) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<MORE_COMM_STATS> TSendTime(
        (syncTypeStr + "Send_" + get_run_identifier(loopName)).c_str(), RNAME);
    galois::CondStatTimer<MORE_COMM_STATS> TRecvTime(
        (syncTypeStr + "Recv_" + get_run_identifier(loopName)).c_str(), RNAME);

    static std::vector<std::vector<uint8_t>> rb;
    static std::vector<MPI_Request> request;

    if (rb.size() == 0) { // create the receive buffers
      TRecvTime.start();
      auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
      rb.resize(numHosts);
      request.resize(numHosts, MPI_REQUEST_NULL);

      for (unsigned h = 1; h < numHosts; ++h) {
        unsigned x = (id + numHosts - h) % numHosts;
        if (nothingToRecv(x, syncType, writeLocation, readLocation))
          continue;

        size_t size =
            (sharedNodes[x].size() * sizeof(typename SyncFnTy::ValTy));
        size += sizeof(size_t);       // vector size
        size += sizeof(DataCommMode); // data mode

        rb[x].resize(size);
      }
      TRecvTime.stop();
    }

    TRecvTime.start();
    sync_mpi_recv_post<writeLocation, readLocation, syncType, SyncFnTy,
                       BitsetFnTy>(loopName, request, rb);
    TRecvTime.stop();

    TSendTime.start();
    if (use_bitset_to_send) {
      sync_mpi_send<writeLocation, readLocation, syncType, SyncFnTy,
                    BitsetFnTy>(loopName);
    } else {
      sync_mpi_send<writeLocation, readLocation, syncType, SyncFnTy,
                    galois::InvalidBitsetFnTy>(loopName);
    }
    TSendTime.stop();

    TRecvTime.start();
    sync_mpi_recv_wait<writeLocation, readLocation, syncType, SyncFnTy,
                       BitsetFnTy>(loopName, request, rb);
    TRecvTime.stop();
  }

  /**
   * Onesided MPI sync
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void syncOnesidedMPI(std::string loopName, bool use_bitset_to_send = true) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<MORE_COMM_STATS> TSendTime(
        (syncTypeStr + "Send_" + get_run_identifier(loopName)).c_str(), RNAME);
    galois::CondStatTimer<MORE_COMM_STATS> TRecvTime(
        (syncTypeStr + "Recv_" + get_run_identifier(loopName)).c_str(), RNAME);

    static std::vector<MPI_Win> window;
    static MPI_Group mpi_access_group;
    static std::vector<std::vector<uint8_t>> rb;

    if (window.size() == 0) { // create the windows
      TRecvTime.start();
      auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
      window.resize(numHosts);
      rb.resize(numHosts);

      uint64_t recv_buffers_size = 0;
      for (unsigned x = 0; x < numHosts; ++x) {
        size_t size =
            (sharedNodes[x].size() * sizeof(typename SyncFnTy::ValTy));
        size += sizeof(size_t);       // vector size
        size += sizeof(DataCommMode); // data mode
        size += sizeof(size_t);       // buffer size
        recv_buffers_size += size;

        rb[x].resize(size);

        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "no_locks", "true");
        MPI_Info_set(info, "same_disp_unit", "true");

        MPI_Win_create(rb[x].data(), size, 1, info, MPI_COMM_WORLD, &window[x]);

        MPI_Info_free(&info);
      }
      auto& net = galois::runtime::getSystemNetworkInterface();
      net.incrementMemUsage(recv_buffers_size);

      for (unsigned h = 1; h < numHosts; ++h) {
        unsigned x = (id + numHosts - h) % numHosts;
        if (nothingToRecv(x, syncType, writeLocation, readLocation))
          continue;
        // exposure group of each window is same as identity group of that
        // window
        MPI_Win_post(mpi_identity_groups[x], 0, window[x]);
      }
      TRecvTime.stop();

      TSendTime.start();
      std::vector<int> access_hosts;
      for (unsigned h = 1; h < numHosts; ++h) {
        unsigned x = (id + h) % numHosts;

        if (nothingToSend(x, syncType, writeLocation, readLocation))
          continue;

        access_hosts.push_back(x);
      }
      MPI_Group world_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);
      // access group for only one window since only one window is accessed
      MPI_Group_incl(world_group, access_hosts.size(), access_hosts.data(),
                     &mpi_access_group);
      TSendTime.stop();
    }

    TSendTime.start();
    if (use_bitset_to_send) {
      sync_mpi_put<writeLocation, readLocation, syncType, SyncFnTy, BitsetFnTy>(
          loopName, mpi_access_group, window);
    } else {
      sync_mpi_put<writeLocation, readLocation, syncType, SyncFnTy,
                   galois::InvalidBitsetFnTy>(loopName, mpi_access_group,
                                              window);
    }
    TSendTime.stop();

    TRecvTime.start();
    sync_mpi_get<writeLocation, readLocation, syncType, SyncFnTy, BitsetFnTy>(
        loopName, window, rb);
    TRecvTime.stop();
  }
#endif

////////////////////////////////////////////////////////////////////////////////
// Higher Level Sync Calls (broadcast/reduce, etc)
////////////////////////////////////////////////////////////////////////////////

  /**
   * Does a reduction of data from mirror nodes to master nodes.
   *
   * @tparam writeLocation Location data is written (src or dst)
   * @tparam readLocation Location data is read (src or dst)
   * @tparam ReduceFnTy reduce sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            typename ReduceFnTy, typename BitsetFnTy, bool async>
  inline void reduce(std::string loopName) {
    std::string timer_str("Reduce_" + get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> TsyncReduce(timer_str.c_str(),
                                                       RNAME);
   
    typedef typename ReduceFnTy::ValTy T;
    typedef typename std::conditional<
        galois::runtime::is_memory_copyable<T>::value,
        galois::PODResizeableArray<T>,
        galois::gstl::Vector<T>>::type
        VecTy;

    TsyncReduce.start();

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    switch (bare_mpi) {
    case noBareMPI:
#endif
      syncSend<writeLocation, readLocation, syncReduce, ReduceFnTy,
                BitsetFnTy, VecTy, async>(loopName);
      syncRecv<writeLocation, readLocation, syncReduce, ReduceFnTy,
                BitsetFnTy, VecTy, async>(loopName);
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
      break;
    case nonBlockingBareMPI:
      syncNonblockingMPI<writeLocation, readLocation, syncReduce, ReduceFnTy,
                           BitsetFnTy>(loopName);
      break;
    case oneSidedBareMPI:
      syncOnesidedMPI<writeLocation, readLocation, syncReduce, ReduceFnTy,
                        BitsetFnTy>(loopName);
      break;
    default:
      GALOIS_DIE("Unsupported bare MPI");
    }
#endif

    TsyncReduce.stop();
  }

  /**
   * Does a broadcast of data from master to mirror nodes.
   *
   * @tparam writeLocation Location data is written (src or dst)
   * @tparam readLocation Location data is read (src or dst)
   * @tparam BroadcastFnTy broadcast sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            typename BroadcastFnTy, typename BitsetFnTy, bool async>
  inline void broadcast(std::string loopName) {
    std::string timer_str("Broadcast_" + get_run_identifier(loopName));
    galois::CondStatTimer<MORE_COMM_STATS> TsyncBroadcast(timer_str.c_str(),
                                                          RNAME);
   
    typedef typename BroadcastFnTy::ValTy T;
    typedef typename std::conditional<
        galois::runtime::is_memory_copyable<T>::value,
        galois::PODResizeableArray<T>,
        galois::gstl::Vector<T>>::type
        VecTy;

    TsyncBroadcast.start();

    bool use_bitset = true;

    if (currentBVFlag != nullptr) {
      if (readLocation == readSource &&
          galois::runtime::src_invalid(*currentBVFlag)) {
        use_bitset     = false;
        *currentBVFlag = BITVECTOR_STATUS::NONE_INVALID;
        currentBVFlag  = nullptr;
      } else if (readLocation == readDestination &&
                 galois::runtime::dst_invalid(*currentBVFlag)) {
        use_bitset     = false;
        *currentBVFlag = BITVECTOR_STATUS::NONE_INVALID;
        currentBVFlag  = nullptr;
      } else if (readLocation == readAny &&
                 *currentBVFlag != BITVECTOR_STATUS::NONE_INVALID) {
        // the bitvector flag being non-null means this call came from
        // sync on demand; sync on demand will NEVER use readAny
        // if location is read Any + one of src or dst is invalid
        GALOIS_DIE("readAny + use of bitvector flag without none_invalid "
                   "should never happen");
      }
    }

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    switch (bare_mpi) {
    case noBareMPI:
#endif
      if (use_bitset) {
        syncSend<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                  BitsetFnTy, VecTy, async>(loopName);
      } else {
        syncSend<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                  galois::InvalidBitsetFnTy, VecTy, async>(loopName);
      }
      syncRecv<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                BitsetFnTy, VecTy, async>(loopName);
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
      break;
    case nonBlockingBareMPI:
      syncNonblockingMPI<writeLocation, readLocation, syncBroadcast,
                           BroadcastFnTy, BitsetFnTy>(loopName, use_bitset);
      break;
    case oneSidedBareMPI:
      syncOnesidedMPI<writeLocation, readLocation, syncBroadcast,
                        BroadcastFnTy, BitsetFnTy>(loopName, use_bitset);
      break;
    default:
      GALOIS_DIE("Unsupported bare MPI");
    }
#endif

    TsyncBroadcast.stop();
  }

  /**
   * Do sync necessary for write source, read source.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_src_to_src(std::string loopName) {
    // do nothing for OEC
    // reduce and broadcast for IEC, CVC, UVC
    if (transposed || isVertexCut) {
      reduce<writeSource, readSource, SyncFnTy, BitsetFnTy, async>(loopName);
      broadcast<writeSource, readSource, SyncFnTy, BitsetFnTy, async>(loopName);
    }
  }

  /**
   * Do sync necessary for write source, read destination.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_src_to_dst(std::string loopName) {
    // only broadcast for OEC
    // only reduce for IEC
    // reduce and broadcast for CVC, UVC
    if (transposed) {
      reduce<writeSource, readDestination, SyncFnTy, BitsetFnTy, async>(loopName);
      if (isVertexCut) {
        broadcast<writeSource, readDestination, SyncFnTy, BitsetFnTy, async>(
            loopName);
      }
    } else {
      if (isVertexCut) {
        reduce<writeSource, readDestination, SyncFnTy, BitsetFnTy, async>(loopName);
      }
      broadcast<writeSource, readDestination, SyncFnTy, BitsetFnTy, async>(
          loopName);
    }
  }

  /**
   * Do sync necessary for write source, read any.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_src_to_any(std::string loopName) {
    // only broadcast for OEC
    // reduce and broadcast for IEC, CVC, UVC
    if (transposed || isVertexCut) {
      reduce<writeSource, readAny, SyncFnTy, BitsetFnTy, async>(loopName);
    }
    broadcast<writeSource, readAny, SyncFnTy, BitsetFnTy, async>(loopName);
  }

  /**
   * Do sync necessary for write dest, read source.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_dst_to_src(std::string loopName) {
    // only reduce for OEC
    // only broadcast for IEC
    // reduce and broadcast for CVC, UVC
    if (transposed) {
      if (isVertexCut) {
        reduce<writeDestination, readSource, SyncFnTy, BitsetFnTy, async>(loopName);
      }
      broadcast<writeDestination, readSource, SyncFnTy, BitsetFnTy, async>(
          loopName);
    } else {
      reduce<writeDestination, readSource, SyncFnTy, BitsetFnTy, async>(loopName);
      if (isVertexCut) {
        broadcast<writeDestination, readSource, SyncFnTy, BitsetFnTy, async>(
            loopName);
      }
    }
  }

  /**
   * Do sync necessary for write dest, read dest.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_dst_to_dst(std::string loopName) {
    // do nothing for IEC
    // reduce and broadcast for OEC, CVC, UVC
    if (!transposed || isVertexCut) {
      reduce<writeDestination, readDestination, SyncFnTy, BitsetFnTy, async>(
          loopName);
      broadcast<writeDestination, readDestination, SyncFnTy, BitsetFnTy, async>(
          loopName);
    }
  }

  /**
   * Do sync necessary for write dest, read any.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_dst_to_any(std::string loopName) {
    // only broadcast for IEC
    // reduce and broadcast for OEC, CVC, UVC
    if (!transposed || isVertexCut) {
      reduce<writeDestination, readAny, SyncFnTy, BitsetFnTy, async>(loopName);
    }
    broadcast<writeDestination, readAny, SyncFnTy, BitsetFnTy, async>(loopName);
  }

  /**
   * Do sync necessary for write any, read src.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_any_to_src(std::string loopName) {
    // only reduce for OEC
    // reduce and broadcast for IEC, CVC, UVC
    reduce<writeAny, readSource, SyncFnTy, BitsetFnTy, async>(loopName);
    if (transposed || isVertexCut) {
      broadcast<writeAny, readSource, SyncFnTy, BitsetFnTy, async>(loopName);
    }
  }

  /**
   * Do sync necessary for write any, read dst.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_any_to_dst(std::string loopName) {
    // only reduce for IEC
    // reduce and broadcast for OEC, CVC, UVC
    reduce<writeAny, readDestination, SyncFnTy, BitsetFnTy, async>(loopName);

    if (!transposed || isVertexCut) {
      broadcast<writeAny, readDestination, SyncFnTy, BitsetFnTy, async>(loopName);
    }
  }

  /**
   * Do sync necessary for write any, read any.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy, bool async>
  inline void sync_any_to_any(std::string loopName) {
    // reduce and broadcast for OEC, IEC, CVC, UVC
    reduce<writeAny, readAny, SyncFnTy, BitsetFnTy, async>(loopName);
    broadcast<writeAny, readAny, SyncFnTy, BitsetFnTy, async>(loopName);
  }

////////////////////////////////////////////////////////////////////////////////
// Public iterface: sync
////////////////////////////////////////////////////////////////////////////////

public:
  /**
   * Main sync call exposed to the user that calls the correct sync function
   * based on provided template arguments. Must provide information through
   * structures on how to do synchronization/which fields to synchronize.
   *
   * @tparam writeLocation Location data is written (src or dst)
   * @tparam readLocation Location data is read (src or dst)
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <WriteLocation writeLocation, ReadLocation readLocation,
            typename SyncFnTy, typename BitsetFnTy = galois::InvalidBitsetFnTy,
            bool async = false>
  inline void sync(std::string loopName) {
    std::string timer_str("Sync_" + loopName + "_" + get_run_identifier());
    galois::StatTimer Tsync(timer_str.c_str(), RNAME);

    Tsync.start();

    if (partitionAgnostic) {
      sync_any_to_any<SyncFnTy, BitsetFnTy, async>(loopName);
    } else {
      if (writeLocation == writeSource) {
        if (readLocation == readSource) {
          sync_src_to_src<SyncFnTy, BitsetFnTy, async>(loopName);
        } else if (readLocation == readDestination) {
          sync_src_to_dst<SyncFnTy, BitsetFnTy, async>(loopName);
        } else { // readAny
          sync_src_to_any<SyncFnTy, BitsetFnTy, async>(loopName);
        }
      } else if (writeLocation == writeDestination) {
        if (readLocation == readSource) {
          sync_dst_to_src<SyncFnTy, BitsetFnTy, async>(loopName);
        } else if (readLocation == readDestination) {
          sync_dst_to_dst<SyncFnTy, BitsetFnTy, async>(loopName);
        } else { // readAny
          sync_dst_to_any<SyncFnTy, BitsetFnTy, async>(loopName);
        }
      } else { // writeAny
        if (readLocation == readSource) {
          sync_any_to_src<SyncFnTy, BitsetFnTy, async>(loopName);
        } else if (readLocation == readDestination) {
          sync_any_to_dst<SyncFnTy, BitsetFnTy, async>(loopName);
        } else { // readAny
          sync_any_to_any<SyncFnTy, BitsetFnTy, async>(loopName);
        }
      }
    }

    Tsync.stop();
  }

////////////////////////////////////////////////////////////////////////////////
// Sync on demand code (unmaintained, may not work)
////////////////////////////////////////////////////////////////////////////////
private:
  /**
   * Generic Sync on demand handler. Should NEVER get to this (hence
   * the galois die).
   */
  template <ReadLocation rl, typename SyncFnTy, typename BitsetFnTy>
  struct SyncOnDemandHandler {
    // note this call function signature is diff. from specialized versions:
    // will cause compile time error if this struct is used (which is what
    // we want)
    void call() { GALOIS_DIE("Invalid read location for sync on demand"); }
  };

  /**
   * Sync on demand handler specialized for read source.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy tells program what data needs to be sync'd
   */
  template <typename SyncFnTy, typename BitsetFnTy>
  struct SyncOnDemandHandler<readSource, SyncFnTy, BitsetFnTy> {
    /**
     * Based on sync flags, handles syncs for cases when you need to read
     * at source
     *
     * @param substrate sync substrate
     * @param fieldFlags the flags structure specifying what needs to be
     * sync'd
     * @param loopName loopname used to name timers
     * @param bvFlag Copy of the bitvector status (valid/invalid at particular
     * locations)
     */
    static inline void call(GluonSubstrate* substrate,
                            galois::runtime::FieldFlags& fieldFlags,
                            std::string loopName,
                            const BITVECTOR_STATUS& bvFlag) {
      if (fieldFlags.src_to_src() && fieldFlags.dst_to_src()) {
        substrate->sync_any_to_src<SyncFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.src_to_src()) {
        substrate->sync_src_to_src<SyncFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.dst_to_src()) {
        substrate->sync_dst_to_src<SyncFnTy, BitsetFnTy>(loopName);
      }

      fieldFlags.clear_read_src();
    }
  };

  /**
   * Sync on demand handler specialized for read destination.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy tells program what data needs to be sync'd
   */
  template <typename SyncFnTy, typename BitsetFnTy>
  struct SyncOnDemandHandler<readDestination, SyncFnTy, BitsetFnTy> {
    /**
     * Based on sync flags, handles syncs for cases when you need to read
     * at destination
     *
     * @param substrate sync substrate
     * @param fieldFlags the flags structure specifying what needs to be
     * sync'd
     * @param loopName loopname used to name timers
     * @param bvFlag Copy of the bitvector status (valid/invalid at particular
     * locations)
     */
    static inline void call(GluonSubstrate* substrate,
                            galois::runtime::FieldFlags& fieldFlags,
                            std::string loopName,
                            const BITVECTOR_STATUS& bvFlag) {
      if (fieldFlags.src_to_dst() && fieldFlags.dst_to_dst()) {
        substrate->sync_any_to_dst<SyncFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.src_to_dst()) {
        substrate->sync_src_to_dst<SyncFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.dst_to_dst()) {
        substrate->sync_dst_to_dst<SyncFnTy, BitsetFnTy>(loopName);
      }

      fieldFlags.clear_read_dst();
    }
  };

  /**
   * Sync on demand handler specialized for read any.
   *
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy tells program what data needs to be sync'd
   */
  template <typename SyncFnTy, typename BitsetFnTy>
  struct SyncOnDemandHandler<readAny, SyncFnTy, BitsetFnTy> {
    /**
     * Based on sync flags, handles syncs for cases when you need to read
     * at both source and destination
     *
     * @param substrate sync substrate
     * @param fieldFlags the flags structure specifying what needs to be
     * sync'd
     * @param loopName loopname used to name timers
     * @param bvFlag Copy of the bitvector status (valid/invalid at particular
     * locations)
     */
    static inline void call(GluonSubstrate* substrate,
                            galois::runtime::FieldFlags& fieldFlags,
                            std::string loopName,
                            const BITVECTOR_STATUS& bvFlag) {
      bool src_write = fieldFlags.src_to_src() || fieldFlags.src_to_dst();
      bool dst_write = fieldFlags.dst_to_src() || fieldFlags.dst_to_dst();

      if (!(src_write && dst_write)) {
        // src or dst write flags aren't set (potentially both are not set),
        // but it's NOT the case that both are set, meaning "any" isn't
        // required in the "from"; can work at granularity of just src
        // write or dst wrte

        if (src_write) {
          if (fieldFlags.src_to_src() && fieldFlags.src_to_dst()) {
            if (bvFlag == BITVECTOR_STATUS::NONE_INVALID) {
              substrate->sync_src_to_any<SyncFnTy, BitsetFnTy>(
                  loopName);
            } else if (galois::runtime::src_invalid(bvFlag)) {
              // src invalid bitset; sync individually so it can be called
              // without bitset
              substrate->sync_src_to_dst<SyncFnTy, BitsetFnTy>(
                  loopName);
              substrate->sync_src_to_src<SyncFnTy, BitsetFnTy>(
                  loopName);
            } else if (galois::runtime::dst_invalid(bvFlag)) {
              // dst invalid bitset; sync individually so it can be called
              // without bitset
              substrate->sync_src_to_src<SyncFnTy, BitsetFnTy>(
                  loopName);
              substrate->sync_src_to_dst<SyncFnTy, BitsetFnTy>(
                  loopName);
            } else {
              GALOIS_DIE("Invalid bitvector flag setting in syncOnDemand");
            }
          } else if (fieldFlags.src_to_src()) {
            substrate->sync_src_to_src<SyncFnTy, BitsetFnTy>(loopName);
          } else { // src to dst is set
            substrate->sync_src_to_dst<SyncFnTy, BitsetFnTy>(loopName);
          }
        } else if (dst_write) {
          if (fieldFlags.dst_to_src() && fieldFlags.dst_to_dst()) {
            if (bvFlag == BITVECTOR_STATUS::NONE_INVALID) {
              substrate->sync_dst_to_any<SyncFnTy, BitsetFnTy>(
                  loopName);
            } else if (galois::runtime::src_invalid(bvFlag)) {
              substrate->sync_dst_to_dst<SyncFnTy, BitsetFnTy>(
                  loopName);
              substrate->sync_dst_to_src<SyncFnTy, BitsetFnTy>(
                  loopName);
            } else if (galois::runtime::dst_invalid(bvFlag)) {
              substrate->sync_dst_to_src<SyncFnTy, BitsetFnTy>(
                  loopName);
              substrate->sync_dst_to_dst<SyncFnTy, BitsetFnTy>(
                  loopName);
            } else {
              GALOIS_DIE("Invalid bitvector flag setting in syncOnDemand");
            }
          } else if (fieldFlags.dst_to_src()) {
            substrate->sync_dst_to_src<SyncFnTy, BitsetFnTy>(loopName);
          } else { // dst to dst is set
            substrate->sync_dst_to_dst<SyncFnTy, BitsetFnTy>(loopName);
          }
        }

        // note the "no flags are set" case will enter into this block
        // as well, and it is correctly handled by doing nothing since
        // both src/dst_write will be false
      } else {
        // it is the case that both src/dst write flags are set, so "any"
        // is required in the "from"; what remains to be determined is
        // the use of src, dst, or any for the destination of the sync
        bool src_read = fieldFlags.src_to_src() || fieldFlags.dst_to_src();
        bool dst_read = fieldFlags.src_to_dst() || fieldFlags.dst_to_dst();

        if (src_read && dst_read) {
          if (bvFlag == BITVECTOR_STATUS::NONE_INVALID) {
            substrate->sync_any_to_any<SyncFnTy, BitsetFnTy>(loopName);
          } else if (galois::runtime::src_invalid(bvFlag)) {
            substrate->sync_any_to_dst<SyncFnTy, BitsetFnTy>(loopName);
            substrate->sync_any_to_src<SyncFnTy, BitsetFnTy>(loopName);
          } else if (galois::runtime::dst_invalid(bvFlag)) {
            substrate->sync_any_to_src<SyncFnTy, BitsetFnTy>(loopName);
            substrate->sync_any_to_dst<SyncFnTy, BitsetFnTy>(loopName);
          } else {
            GALOIS_DIE("Invalid bitvector flag setting in syncOnDemand");
          }
        } else if (src_read) {
          substrate->sync_any_to_src<SyncFnTy, BitsetFnTy>(loopName);
        } else { // dst_read
          substrate->sync_any_to_dst<SyncFnTy, BitsetFnTy>(loopName);
        }
      }

      fieldFlags.clear_read_src();
      fieldFlags.clear_read_dst();
    }
  };

////////////////////////////////////////////////////////////////////////////////
// GPU marshaling
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
private:
  using GraphNode = typename GraphTy::GraphNode;
  using edge_iterator = typename GraphTy::edge_iterator;
  using EdgeTy = typename GraphTy::EdgeType;

  // Code that handles getting the graph onto the GPU
  template <bool isVoidType,
            typename std::enable_if<isVoidType>::type* = nullptr>
  inline void setMarshalEdge(MarshalGraph& m, const size_t index,
                             const edge_iterator& e) {
    // do nothing
  }

  template <bool isVoidType,
            typename std::enable_if<!isVoidType>::type* = nullptr>
  inline void setMarshalEdge(MarshalGraph& m, const size_t index,
                             const edge_iterator& e) {
    m.edge_data[index] = userGraph.getEdgeData(e);
  }

public:
  void getMarshalGraph(MarshalGraph& m) {
    m.nnodes = userGraph.size();
    m.nedges = userGraph.sizeEdges();
    m.numOwned          = userGraph.numMasters();
    // Assumption: master occurs at beginning in contiguous range
    m.beginMaster       = 0;
    m.numNodesWithEdges = userGraph.getNumNodesWithEdges();
    m.id                = id;
    m.numHosts          = numHosts;
    m.row_start         = (index_type*)calloc(m.nnodes + 1, sizeof(index_type));
    m.edge_dst          = (index_type*)calloc(m.nedges, sizeof(index_type));
    m.node_data         = (index_type*)calloc(m.nnodes, sizeof(node_data_type));

    // TODO deal with edgety
    if (std::is_void<EdgeTy>::value) {
      m.edge_data = NULL;
    } else {
      if (!std::is_same<EdgeTy, edge_data_type>::value) {
        galois::gWarn("Edge data type mismatch between CPU and GPU\n");
      }
      m.edge_data = (edge_data_type*)calloc(m.nedges, sizeof(edge_data_type));
    }

    galois::do_all(
      // TODO not using thread ranges, can be optimized if I can iterate
      // directly over userGraph
      galois::iterate(userGraph.allNodesRange()),
      [&](const GraphNode& nodeID) {
        // initialize node_data with localID-to-globalID mapping
        m.node_data[nodeID] = userGraph.getGID(nodeID);
        m.row_start[nodeID] = *(userGraph.edge_begin(nodeID));
        for (auto e = userGraph.edge_begin(nodeID);
             e != userGraph.edge_end(nodeID);
             e++) {
          auto edgeID = *e;
          setMarshalEdge<std::is_void<EdgeTy>::value>(m, edgeID, e);
          m.edge_dst[edgeID] = userGraph.getEdgeDst(e);
        }
      },
      galois::steal()
    );

    m.row_start[m.nnodes] = m.nedges;

    ////// TODO

    // copy memoization meta-data
    m.num_master_nodes =
        (unsigned int*)calloc(masterNodes.size(), sizeof(unsigned int));
    ;
    m.master_nodes =
        (unsigned int**)calloc(masterNodes.size(), sizeof(unsigned int*));
    ;

    for (uint32_t h = 0; h < masterNodes.size(); ++h) {
      m.num_master_nodes[h] = masterNodes[h].size();

      if (masterNodes[h].size() > 0) {
        m.master_nodes[h] =
            (unsigned int*)calloc(masterNodes[h].size(), sizeof(unsigned int));
        ;
        std::copy(masterNodes[h].begin(), masterNodes[h].end(),
                  m.master_nodes[h]);
      } else {
        m.master_nodes[h] = NULL;
      }
    }

    m.num_mirror_nodes =
        (unsigned int*)calloc(mirrorNodes.size(), sizeof(unsigned int));
    ;
    m.mirror_nodes =
        (unsigned int**)calloc(mirrorNodes.size(), sizeof(unsigned int*));
    ;
    for (uint32_t h = 0; h < mirrorNodes.size(); ++h) {
      m.num_mirror_nodes[h] = mirrorNodes[h].size();

      if (mirrorNodes[h].size() > 0) {
        m.mirror_nodes[h] =
            (unsigned int*)calloc(mirrorNodes[h].size(), sizeof(unsigned int));
        ;
        std::copy(mirrorNodes[h].begin(), mirrorNodes[h].end(),
                  m.mirror_nodes[h]);
      } else {
        m.mirror_nodes[h] = NULL;
      }
    }

    // user needs to provide method of freeing up graph (it can do nothing
    // if they wish)
    userGraph.deallocate();
  }
#endif // het galois def

////////////////////////////////////////////////////////////////////////////////
// Public sync interface
////////////////////////////////////////////////////////////////////////////////

public:
  /**
   * Given a structure that contains flags signifying what needs to be
   * synchronized, syncOnDemand will synchronize what is necessary based
   * on the read location of the * field.
   *
   * @tparam readLocation Location in which field will need to be read
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct which holds a bitset which can be used
   * to control synchronization at a more fine grain level
   * @param fieldFlags structure for field you are syncing
   * @param loopName Name of loop this sync is for for naming timers
   */
  template <ReadLocation readLocation, typename SyncFnTy,
            typename BitsetFnTy = galois::InvalidBitsetFnTy>
  inline void syncOnDemand(galois::runtime::FieldFlags& fieldFlags,
                           std::string loopName) {
    std::string timer_str("Sync_" + get_run_identifier(loopName));
    galois::StatTimer Tsync(timer_str.c_str(), RNAME);
    Tsync.start();

    currentBVFlag = &(fieldFlags.bitvectorStatus);

    // call a template-specialized function depending on the read location
    SyncOnDemandHandler<readLocation, SyncFnTy,
                        BitsetFnTy>::call(this, fieldFlags, loopName,
                                          *currentBVFlag);

    currentBVFlag = nullptr;

    Tsync.stop();
  }

////////////////////////////////////////////////////////////////////////////////
// Metadata settings/getters
////////////////////////////////////////////////////////////////////////////////
  /**
   * Set the run number.
   *
   * @param runNum Number to set the run to
   */
  inline void set_num_run(const uint32_t runNum) { num_run = runNum; }

  /**
   * Get the set run number.
   *
   * @returns The set run number saved in the graph
   */
  inline uint32_t get_run_num() const { return num_run; }

  /**
   * Set the round number for use in the run identifier.
   *
   * @param round round number to set to
   */
  inline void set_num_round(const uint32_t round) { num_round = round; }

  /**
   * Get a run identifier using the set run and set round.
   *
   * @returns a string run identifier
   * @deprecated We want to move away from calling this by itself; use ones
   * that take an argument; will be removed once we eliminate all instances
   * of its use from code
   */
  inline std::string get_run_identifier() const {
#if DIST_PER_ROUND_TIMER
    return std::string(std::to_string(num_run) + "_" +
                       std::to_string(num_round));
#else
    return std::string(std::to_string(num_run));
#endif
  }

  /**
   * Get a run identifier using the set run and set round and
   * append to the passed in string.
   *
   * @param loop_name String to append the run identifier
   * @returns String with run identifier appended to passed in loop name
   */
  inline std::string get_run_identifier(std::string loop_name) const {
#if DIST_PER_ROUND_TIMER
    return std::string(std::string(loop_name) + "_" + std::to_string(num_run) +
                       "_" + std::to_string(num_round));
#else
    return std::string(std::string(loop_name) + "_" + std::to_string(num_run));
#endif
  }

  /**
   * Get a run identifier using the set run and set round and
   * append to the passed in string in addition to the number identifier passed
   * in.
   *
   * @param loop_name String to append the run identifier
   * @param alterID another ID with which to add to the timer name.
   *
   * @returns String with run identifier appended to passed in loop name +
   * alterID
   */
  inline std::string get_run_identifier(std::string loop_name,
                                        unsigned alterID) const {
#if DIST_PER_ROUND_TIMER
    return std::string(std::string(loop_name) + "_" + std::to_string(alterID) +
                       "_" + std::to_string(num_run) + "_" +
                       std::to_string(num_round));
#else
    return std::string(std::string(loop_name) + "_" + std::to_string(alterID) +
                       "_" + std::to_string(num_run));
#endif
  }

  /**
   * Given a sync structure, reset the field specified by the structure
   * to the 0 of the reduction on mirrors.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done
   */
  template <typename FnTy>
  void reset_mirrorField() {
    // TODO make sure this is correct still
    auto mirrorRanges = userGraph.getMirrorRanges();
    for (auto r : mirrorRanges) {
      if (r.first == r.second) continue;
      assert(r.first < r.second);

      // GPU call
      bool batch_succeeded = FnTy::reset_batch(r.first, r.second - 1);

      // CPU always enters this block
      if (!batch_succeeded) {
        galois::do_all(
            galois::iterate(r.first, r.second),
            [&](uint32_t lid) {
              FnTy::reset(lid, userGraph.getData(lid));
            },
            galois::no_stats(),
            galois::loopname(get_run_identifier("RESET:MIRRORS").c_str()));
      }
    }
  }


////////////////////////////////////////////////////////////////////////////////
// Checkpointing code for graph
////////////////////////////////////////////////////////////////////////////////

// @todo Checkpointing code needs updates to make it work.
#ifdef __GALOIS_CHECKPOINT__
///*
// * Headers for boost serialization
// */
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/serialization/split_member.hpp>
//#include <boost/serialization/binary_object.hpp>
//#include <boost/serialization/serialization.hpp>
//#include <boost/serialization/vector.hpp>
//#include <boost/serialization/unordered_map.hpp>
//
//public:
//  /**
//   * Checkpoint the complete structure on the node to disk
//   */
//  void checkpointSaveNodeData(std::string checkpointFileName = "checkpoint") {
//    using namespace boost::archive;
//    galois::StatTimer TimerSaveCheckPoint(
//        get_run_identifier("TimerSaveCheckpoint").c_str(), RNAME);
//
//    TimerSaveCheckPoint.start();
//    std::string checkpointFileName_local =
//        checkpointFileName + "_" + std::to_string(id);
//
//    std::ofstream outputStream(checkpointFileName_local, std::ios::binary);
//    if (!outputStream.is_open()) {
//      galois::gPrint("ERROR: Could not open ", checkpointFileName_local,
//                     " to save checkpoint!!!\n");
//    }
//    galois::gPrint("[", id,
//                   "] Saving local checkpoint to :", checkpointFileName_local,
//                   "\n");
//
//    boost::archive::binary_oarchive ar(outputStream, boost::archive::no_header);
//
//    // TODO handle this with CuSP
//    userGraph.serializeNodeData(ar);
//
//    std::string statSendBytes_str("CheckpointBytesTotal");
//    constexpr static const char* const RREGION = "RECOVERY";
//    size_t cp_size                             = outputStream.tellp();
//    galois::runtime::reportStat_Tsum(RREGION, statSendBytes_str, cp_size);
//
//    outputStream.flush();
//    outputStream.close();
//    TimerSaveCheckPoint.stop();
//  }
//
//  /**
//   * Load checkpointed data from disk.
//   */
//  void checkpointApplyNodeData(std::string checkpointFileName = "checkpoint") {
//    using namespace boost::archive;
//    galois::StatTimer TimerApplyCheckPoint(
//        get_run_identifier("TimerApplyCheckpoint").c_str(), RNAME);
//
//    TimerApplyCheckPoint.start();
//    std::string checkpointFileName_local =
//        checkpointFileName + "_" + std::to_string(id);
//
//    std::ifstream inputStream(checkpointFileName_local, std::ios::binary);
//
//    if (!inputStream.is_open()) {
//      galois::gPrint("ERROR: Could not open ", checkpointFileName_local,
//                     " to read checkpoint!!!\n");
//    }
//    galois::gPrint("[", id, "] reading local checkpoint from: ",
//                   checkpointFileName_local, "\n");
//
//    boost::archive::binary_iarchive ar(inputStream, boost::archive::no_header);
//
//    // TODO handle this with CuSP
//    userGraph.deSerializeNodeData(ar);
//
//    inputStream.close();
//    TimerApplyCheckPoint.stop();
//  }
#endif
};

template <typename GraphTy>
constexpr const char* const galois::graphs::GluonSubstrate<GraphTy>::RNAME;
} // end namespace graphs
} // end namespace galois

#endif // header guard
