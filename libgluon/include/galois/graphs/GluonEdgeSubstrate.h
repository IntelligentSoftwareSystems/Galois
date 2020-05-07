/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
 * @file GluonEdgeSubstrate.h
 *
 * Contains the implementation for GluonEdgeSubstrate.
 */

// TODO merge with GluonSubstrate; way too much code duplication

#ifndef _GALOIS_GLUONEDGESUB_H_
#define _GALOIS_GLUONEDGESUB_H_

#include <unordered_map>
#include <fstream>

#include "galois/runtime/GlobalObj.h"
#include "galois/runtime/DistStats.h"
#include "galois/runtime/SyncStructures.h"
#include "galois/runtime/DataCommMode.h"
#include "galois/DynamicBitset.h"

#ifdef GALOIS_ENABLE_GPU
#include "galois/cuda/EdgeHostDecls.h"
#endif

#include "galois/runtime/BareMPI.h"

// TODO make not global
//! Specifies what format to send metadata in
extern DataCommMode enforcedDataMode;

#ifdef GALOIS_USE_BARE_MPI
//! bare_mpi type to use; see options in runtime/BareMPI.h
BareMPI bare_mpi = BareMPI::noBareMPI;
#endif

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
class GluonEdgeSubstrate : public galois::runtime::GlobalObject {
private:
  //! Synchronization type
  enum SyncType {
    syncReduce,   //!< Reduction sync
    syncBroadcast //!< Broadcast sync
  };

  //! Graph name used for printing things
  constexpr static const char* const RNAME = "GluonEdges";

  //! The graph to handle communication for
  GraphTy& userGraph;
  const unsigned id; //!< Copy of net.ID, which is the ID of the machine.
  DataCommMode substrateDataMode; //!< datamode to enforce
  const uint32_t
      numHosts;     //!< Copy of net.Num, which is the total number of machines
  uint32_t num_run; //!< Keep track of number of runs.
  uint32_t num_round; //!< Keep track of number of rounds.

  // memoization optimization
  //! Master edges on different hosts. For broadcast;
  std::vector<std::vector<size_t>> masterEdges;
  //! Mirror edges on different hosts. For reduce; comes from the user graph
  //! during initialization (we expect user to give to us)
  std::vector<std::vector<size_t>>& mirrorEdges;
  //! Maximum size of master or mirror edges on different hosts
  size_t maxSharedSize;

#ifdef GALOIS_USE_BARE_MPI
  std::vector<MPI_Group> mpi_identity_groups;
#endif
  // Used for efficient comms
  galois::DynamicBitSet syncBitset;
  galois::PODResizeableArray<unsigned int> syncOffsets;

  void reset_bitset(SyncType syncType,
                    void (*bitset_reset_range)(size_t, size_t)) {
    if (userGraph.sizeEdges() > 0) {
      bitset_reset_range(0, userGraph.sizeEdges() - 1);
    }
  }

  //! Increments evilPhase, a phase counter used by communication.
  void inline incrementEvilPhase() {
    ++galois::runtime::evilPhase;
    // limit defined by MPI or LCI
    if (galois::runtime::evilPhase >=
        uint32_t{std::numeric_limits<int16_t>::max()}) {
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

    // send off the mirror edges
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id)
        continue;

      galois::runtime::SendBuffer b;
      gSerialize(b, mirrorEdges[x]);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    // receive the mirror edges
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id)
        continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      galois::runtime::gDeserialize(p->second, masterEdges[p->first]);
    }
    incrementEvilPhase();
  }

  /**
   * Send statistics about master/mirror edges to each host, and
   * report the statistics.
   */
  void sendInfoToHost() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    uint64_t totalMirrorEdges =
        userGraph.sizeEdges() - userGraph.numOwnedEdges();
    uint64_t totalOwnedEdges = userGraph.numOwnedEdges();

    // send info to host
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id)
        continue;

      galois::runtime::SendBuffer b;
      gSerialize(b, totalMirrorEdges, totalOwnedEdges);
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

      uint64_t totalMirrorFromOther;
      uint64_t totalOwnedFromOther;
      galois::runtime::gDeserialize(p->second, totalMirrorFromOther,
                                    totalOwnedFromOther);
      totalMirrorEdges += totalMirrorFromOther;
      totalOwnedEdges += totalOwnedFromOther;
    }
    incrementEvilPhase();

    assert(userGraph.globalEdges() == totalOwnedEdges);

    // report stats
    if (net.ID == 0) {
      reportProxyStats(totalMirrorEdges, totalOwnedEdges);
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
    // convert the global ids stored in the master/mirror edges arrays to local
    // ids
    // TODO: use 32-bit distinct vectors for masters and mirrors from here on
    for (uint32_t h = 0; h < masterEdges.size(); ++h) {
      galois::do_all(
          galois::iterate(size_t{0}, masterEdges[h].size()),
          [&](size_t n) {
            masterEdges[h][n] = userGraph.getEdgeLID(masterEdges[h][n]);
          },
#if GALOIS_COMM_STATS
          galois::loopname(get_run_identifier("MasterEdges").c_str()),
#endif
          galois::no_stats());
    }

    for (uint32_t h = 0; h < mirrorEdges.size(); ++h) {
      galois::do_all(
          galois::iterate(size_t{0}, mirrorEdges[h].size()),
          [&](size_t n) {
            mirrorEdges[h][n] = userGraph.getEdgeLID(mirrorEdges[h][n]);
          },
#if GALOIS_COMM_STATS
          galois::loopname(get_run_identifier("MirrorEdges").c_str()),
#endif
          galois::no_stats());
    }

    Tcomm_setup.stop();

    maxSharedSize = 0;
    // report masters/mirrors to/from other hosts as statistics
    for (auto x = 0U; x < masterEdges.size(); ++x) {
      if (x == id)
        continue;
      std::string master_edges_str =
          "MasterEdgesFrom_" + std::to_string(id) + "_To_" + std::to_string(x);
      galois::runtime::reportStatCond_Tsum<MORE_DIST_STATS>(
          RNAME, master_edges_str, masterEdges[x].size());
      if (masterEdges[x].size() > maxSharedSize) {
        maxSharedSize = masterEdges[x].size();
      }
    }

    for (auto x = 0U; x < mirrorEdges.size(); ++x) {
      if (x == id)
        continue;
      std::string mirror_edges_str =
          "MirrorEdgesFrom_" + std::to_string(x) + "_To_" + std::to_string(id);
      galois::runtime::reportStatCond_Tsum<MORE_DIST_STATS>(
          RNAME, mirror_edges_str, mirrorEdges[x].size());
      if (mirrorEdges[x].size() > maxSharedSize) {
        maxSharedSize = mirrorEdges[x].size();
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
   * @param totalMirrorEdges number of mirror edges on all hosts
   * @param totalOwnedEdges number of "owned" edges on all hosts
   */
  void reportProxyStats(uint64_t totalMirrorEdges, uint64_t totalOwnedEdges) {
    float replication_factor =
        (float)(totalMirrorEdges + userGraph.globalEdges()) /
        (float)userGraph.globalEdges();
    galois::runtime::reportStat_Single(RNAME, "ReplicationFactorEdges",
                                       replication_factor);
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(
        RNAME, "TotalGlobalMirrorEdges", totalMirrorEdges);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Initializers
  ////////////////////////////////////////////////////////////////////////////////
  /**
   * Initalize MPI related things. The MPI layer itself should have been
   * initialized when the network interface was initiailized.
   */
  void initBareMPI() {
#ifdef GALOIS_USE_BARE_MPI
    if (bare_mpi == noBareMPI)
      return;

#ifdef GALOIS_USE_LCI
    // sanity check of ranks
    int taskRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);
    if ((unsigned)taskRank != id)
      GALOIS_DIE("Mismatch in MPI rank");
    int numTasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    if ((unsigned)numTasks != numHosts)
      GALOIS_DIE("Mismatch in MPI rank");
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
  GluonEdgeSubstrate() = delete;

  /**
   * Constructor for GluonEdgeSubstrate. Initializes metadata fields.
   *
   * @param host host number that this graph resides on
   * @param numHosts total number of hosts in the currently executing program
   */
  GluonEdgeSubstrate(GraphTy& _userGraph, unsigned host, unsigned numHosts,
                     bool doNothing                  = false,
                     DataCommMode _substrateDataMode = DataCommMode::noData)
      : galois::runtime::GlobalObject(this), userGraph(_userGraph), id(host),
        substrateDataMode(_substrateDataMode), numHosts(numHosts), num_run(0),
        num_round(0), mirrorEdges(userGraph.getMirrorEdges()) {
    if (!doNothing) {
      galois::StatTimer edgeSubstrateSetupTimer(
          "GluonEdgeSubstrateConstructTime", RNAME);
      edgeSubstrateSetupTimer.start();

      // set global
      enforcedDataMode = _substrateDataMode;

      initBareMPI();
      // master setup from mirrors done by setupCommunication call
      masterEdges.resize(numHosts);

      // setup proxy communication
      galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct_comm(
          "GraphCommSetupTime", RNAME);
      Tgraph_construct_comm.start();
      setupCommunication();
      Tgraph_construct_comm.stop();

      edgeSubstrateSetupTimer.stop();
    }
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
    galois::CondStatTimer<GALOIS_COMM_STATS> Toffsets(offsets_timer_str.c_str(),
                                                    RNAME);

    Toffsets.start();

    auto activeThreads = galois::getActiveThreads();
    std::vector<unsigned int> t_prefix_bit_counts(activeThreads);

    // count how many bits are set on each thread
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      // TODO use block_range instead
      unsigned int block_size = bitset_comm.size() / nthreads;
      if ((bitset_comm.size() % nthreads) > 0)
        ++block_size;
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
   * edges.
   *
   * @tparam FnTy structure that specifies how synchronization is to be done;
   * only used to get the size of the type being synchronized in this function
   * @tparam syncType type of synchronization this function is being called
   * for; only used to name a timer
   *
   * @param loopName loopname used to name the timer for the function
   * @param indices A vector that contains the local ids of the edges that
   * you want to potentially synchronize
   * @param bitset_compute Contains the full bitset of all edges in this
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
      // determine which local edges in the indices array need to be
      // sychronized
      galois::do_all(
          galois::iterate(size_t{0}, indices.size()),
          [&](size_t n) {
            // assumes each lid is unique as test is not thread safe
            size_t lid = indices[n];
            if (bitset_compute.test(lid)) {
              bitset_comm.set(n);
            }
          },
#if GALOIS_COMM_STATS
          galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
          galois::no_stats());

      // get the number of set bits and the offsets into the comm bitset
      getOffsetsFromBitset<syncType>(loopName, bitset_comm, offsets,
                                     bit_set_count);
    }

    data_mode =
        get_data_mode<typename FnTy::ValTy>(bit_set_count, indices.size());
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Local to global ID conversion
  ////////////////////////////////////////////////////////////////////////////////
  /**
   * Converts LIDs of edges we are interested in into GIDs.
   *
   * @tparam syncType either reduce or broadcast; only used to name the timer
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of edges that we are interested in
   * @param offsets INPUT/OUTPUT holds offsets into "indices" that we should
   * use; after function completion, holds global ids of edges we are interested
   * in
   */
  template <SyncType syncType>
  void convertLIDToGID(const std::string& loopName,
                       const std::vector<size_t>& indices,
                       galois::PODResizeableArray<unsigned int>& offsets) {
    galois::gWarn("LID to GID edge conversion is extremely inefficient at the "
                  "moment!");
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string doall_str(syncTypeStr + "_LID2GID_" +
                          get_run_identifier(loopName));
    galois::do_all(
        galois::iterate(size_t{0}, offsets.size()),
        [&](size_t n) {
          offsets[n] =
              static_cast<uint32_t>(userGraph.getEdgeGID(indices[offsets[n]]));
        },
#if GALOIS_COMM_STATS
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
    galois::gWarn("WARNING: convert GID to LID used in sync call (not "
                  "optimized)");
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string doall_str(syncTypeStr + "_GID2LID_" +
                          get_run_identifier(loopName));

    galois::do_all(
        galois::iterate(size_t{0}, offsets.size()),
        [&](size_t n) { offsets[n] = userGraph.getEdgeLID(offsets[n]); },
#if GALOIS_COMM_STATS
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
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, bool async,
      typename std::enable_if<!BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  void getSendBuffer(std::string loopName, unsigned x,
                     galois::runtime::SendBuffer& b) {
    auto& sharedEdges = (syncType == syncReduce) ? mirrorEdges : masterEdges;

    if (BitsetFnTy::is_valid()) {
      syncExtract<syncType, SyncFnTy, BitsetFnTy, async>(loopName, x,
                                                         sharedEdges[x], b);
    } else {
      syncExtract<syncType, SyncFnTy, async>(loopName, x, sharedEdges[x], b);
    }

    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string statSendBytes_str(syncTypeStr + "SendBytes_" +
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
   * @param indices list of all edges that we are potentially interested in
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
    galois::CondStatTimer<GALOIS_COMM_STATS> Tserialize(
        serialize_timer_str.c_str(), RNAME);
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
   * @param bit_set_comm holds the bitset representing changed edges after
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
                          galois::DynamicBitSet& bit_set_comm,
                          size_t& buf_start, size_t& retval, VecType& val_vec) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string serialize_timer_str(syncTypeStr + "DeserializeMessage_" +
                                    get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Tdeserialize(
        serialize_timer_str.c_str(), RNAME);
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
  // Requirement: For all X and Y,
  // On X, nothingToSend(Y) <=> On Y, nothingToRecv(X)
  /**
   * Determine if we have anything that we need to send to a particular host
   *
   * @param host Host number that we may or may not send to
   * @param syncType Synchronization type to determine which edges on a
   * host need to be considered
   * @returns true if there is nothing to send to a host, false otherwise
   */
  bool nothingToSend(unsigned host, SyncType syncType) {
    auto& sharedEdges = (syncType == syncReduce) ? mirrorEdges : masterEdges;
    return (sharedEdges[host].size() == 0);
  }

  /**
   * Determine if we have anything that we need to receive from a particular
   * host
   *
   * @param host Host number that we may or may not receive from
   * @param syncType Synchronization type to determine which edges on a
   * host need to be considered
   * @returns true if there is nothing to receive from a host, false otherwise
   */
  bool nothingToRecv(unsigned host, SyncType syncType) {
    auto& sharedEdges = (syncType == syncReduce) ? masterEdges : mirrorEdges;
    return (sharedEdges[host].size() == 0);
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
   * @param totalToSend Total amount of edges that are potentially sent (not
   * necessarily all nodees will be sent)
   * @param bitSetCount Number of edges that will actually be sent
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
  // Extract data from edges (for reduce and broadcast)
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
      auto val = FnTy::extract(lid, userGraph.getEdgeData(lid));
      FnTy::reset(lid, userGraph.getEdgeData(lid));
      return val;
    } else {
      return FnTy::extract(lid, userGraph.getEdgeData(lid));
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
      auto val = FnTy::extract(lid, userGraph.getEdgeData(lid), vecIndex);
      FnTy::reset(lid, userGraph.getEdgeData(lid), vecIndex);
      return val;
    } else {
      return FnTy::extract(lid, userGraph.getEdgeData(lid), vecIndex);
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
   * @param indices Local ids of edges that we are interested in
   * @param size Number of elements to extract
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param val_vec OUTPUT: holds the extracted data
   * @param start Offset into val_vec to start saving data to
   */
  template <typename FnTy, SyncType syncType, bool identity_offsets = false,
            bool parallelize = true>
  void extractSubset(const std::string& loopName,
                     const std::vector<size_t>& indices, size_t size,
                     const galois::PODResizeableArray<unsigned int>& offsets,
                     galois::PODResizeableArray<typename FnTy::ValTy>& val_vec,
                     size_t start = 0) {
    if (parallelize) {
      std::string syncTypeStr =
          (syncType == syncReduce) ? "Reduce" : "Broadcast";
      std::string doall_str(syncTypeStr + "ExtractVal_" + loopName);

      galois::do_all(
          galois::iterate(start, start + size),
          [&](unsigned int n) {
            unsigned int offset;
            if (identity_offsets)
              offset = n;
            else
              offset = offsets[n];
            size_t lid         = indices[offset];
            val_vec[n - start] = extractWrapper<FnTy, syncType>(lid);
          },
#if GALOIS_COMM_STATS
          galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
          galois::no_stats());
    } else { // non-parallel version
      for (unsigned n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets)
          offset = n;
        else
          offset = offsets[n];

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
   * @param indices Local ids of edges that we are interested in
   * @param size Number of elements to extract
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param val_vec OUTPUT: holds the extracted data
   * @param vecIndex which element of the vector to extract from node
   * @param start Offset into val_vec to start saving data to
   */
  // TODO find a better way to have this variant without code duplication
  template <typename FnTy, SyncType syncType, bool identity_offsets = false,
            bool parallelize = true, bool vecSync = false,
            typename std::enable_if<vecSync>::type* = nullptr>
  void extractSubset(const std::string& loopName,
                     const std::vector<size_t>& indices, size_t size,
                     const galois::PODResizeableArray<unsigned int>& offsets,
                     galois::PODResizeableArray<typename FnTy::ValTy>& val_vec,
                     unsigned vecIndex, size_t start = 0) {
    val_vec.resize(size); // resize val vec for this vecIndex

    if (parallelize) {
      std::string syncTypeStr =
          (syncType == syncReduce) ? "Reduce" : "Broadcast";
      std::string doall_str(syncTypeStr + "ExtractValVector_" + loopName);

      galois::do_all(
          galois::iterate(start, start + size),
          [&](unsigned int n) {
            unsigned int offset;
            if (identity_offsets)
              offset = n;
            else
              offset = offsets[n];
            size_t lid         = indices[offset];
            val_vec[n - start] = extractWrapper<FnTy, syncType>(lid, vecIndex);
          },
#if GALOIS_COMM_STATS
          galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
          galois::no_stats());
    } else { // non-parallel version
      for (unsigned n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets)
          offset = n;
        else
          offset = offsets[n];
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
   * @param indices Local ids of edges that we are interested in
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

      galois::do_all(
          galois::iterate(start, start + size),
          [&](unsigned int n) {
            unsigned int offset;
            if (identity_offsets)
              offset = n;
            else
              offset = offsets[n];

            size_t lid = indices[offset];
            gSerializeLazy(b, lseq, n - start,
                           extractWrapper<FnTy, syncType>(lid));
          },
#if GALOIS_COMM_STATS
          galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
          galois::no_stats());
    } else {
      for (unsigned int n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets)
          offset = n;
        else
          offset = offsets[n];
        size_t lid = indices[offset];
        gSerializeLazy(b, lseq, n - start, extractWrapper<FnTy, syncType>(lid));
      }
    }
  }

  /**
   * GPU wrap function: extracts data from edges and resets them to the
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
   * GPU wrap function: extracts data from edges and resets them to the
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
   * @param bit_set_compute bitset indicating which edges have changed; updated
   * if reduction causes a change
   */
  template <typename FnTy, SyncType syncType, bool async>
  inline void setWrapper(size_t lid, typename FnTy::ValTy val,
                         galois::DynamicBitSet& bit_set_compute) {
    if (syncType == syncReduce) {
      if (FnTy::reduce(lid, userGraph.getEdgeData(lid), val)) {
        if (bit_set_compute.size() != 0) {
          bit_set_compute.set(lid);
        }
      }
    } else {
      if (async) {
        FnTy::reduce(lid, userGraph.getEdgeData(lid), val);
      } else {
        // uint64_t edgeSource =
        // userGraph.getGID(userGraph.findSourceFromEdge(lid)); if (val !=
        // userGraph.getHostID(edgeSource)) {
        //  GALOIS_DIE(galois::runtime::getSystemNetworkInterface().ID, " ",
        //  edgeSource, " ", val, " ", userGraph.getHostID(edgeSource));

        //  assert(val == userGraph.getHostID(edgeSource));
        //}

        // galois::gPrint("[", galois::runtime::getSystemNetworkInterface().ID ,
        //               "] broadcast, val is ", val, " edge srouce ",
        //               userGraph.getGID(userGraph.findSourceFromEdge(lid)),
        //               "\n");
        FnTy::setVal(lid, userGraph.getEdgeData(lid), val);
        assert(FnTy::extract(lid, userGraph.getEdgeData(lid)) == val);
      }
    }
  }

  /**
   * Given data received from another host and information on which edges
   * to update, do the reduce/set of the received data to update local edges.
   *
   * Complement function, in some sense, of extractSubset.
   *
   * @tparam VecTy type of indices variable
   * @tparam FnTy structure that specifies how synchronization is to be done
   * @tparam SyncType Reduce or broadcast
   * @tparam identity_offsets If this is true, then ignore the offsets
   * array and just grab directly from indices (i.e. don't pick out
   * particular elements, just grab contiguous chunk)
   * @tparam parallelize True if updates to edges are to be parallelized
   *
   * @param loopName name of loop used to name timer
   * @param indices Local ids of edges that we are interested in
   * @param size Number of elements to set
   * @param offsets Holds offsets into "indices" of the data that we are
   * interested in
   * @param val_vec holds data we will use to set
   * @param bit_set_compute bitset indicating which edges have changed
   * @param start Offset into val_vec to start saving data to
   */
  template <typename VecTy, typename FnTy, SyncType syncType, bool async,
            bool identity_offsets = false, bool parallelize = true>
  void setSubset(const std::string& loopName, const VecTy& indices, size_t size,
                 const galois::PODResizeableArray<unsigned int>& offsets,
                 galois::PODResizeableArray<typename FnTy::ValTy>& val_vec,
                 galois::DynamicBitSet& bit_set_compute, size_t start = 0) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string doall_str(syncTypeStr + "SetVal_" +
                          get_run_identifier(loopName));

    if (parallelize) {
      galois::do_all(
          galois::iterate(start, start + size),
          [&](unsigned int n) {
            unsigned int offset;
            if (identity_offsets)
              offset = n;
            else
              offset = offsets[n];
            auto lid = indices[offset];
            setWrapper<FnTy, syncType, async>(lid, val_vec[n - start],
                                              bit_set_compute);
          },
#if GALOIS_COMM_STATS
          galois::loopname(get_run_identifier(doall_str).c_str()),
#endif
          galois::no_stats());
    } else {
      for (unsigned int n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets)
          offset = n;
        else
          offset = offsets[n];
        auto lid = indices[offset];
        setWrapper<FnTy, syncType, async>(lid, val_vec[n - start],
                                          bit_set_compute);
      }
    }
  }

  /**
   * GPU wrapper function to reduce multiple edges at once.
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
   * GPU wrapper function to reduce multiple edges at once. More detailed
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
      return FnTy::reduce_batch(x, b.getVec().data() + b.getOffset(),
                                data_mode);
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
   * @param indices Vector that contains node ids of edges that we will
   * potentially send things to
   * @param b OUTPUT: buffer that will be sent over the network; contains data
   * based on set bits in bitset
   */
  template <SyncType syncType, typename SyncFnTy, bool async,
            typename std::enable_if<galois::runtime::is_memory_copyable<
                typename SyncFnTy::ValTy>::value>::type* = nullptr>
  void syncExtract(std::string loopName, unsigned from_id,
                   std::vector<size_t>& indices,
                   galois::runtime::SendBuffer& b) {
    uint32_t num = indices.size();
    static galois::PODResizeableArray<typename SyncFnTy::ValTy>
        val_vec; // sometimes wasteful
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string extract_timer_str(syncTypeStr + "Extract_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Textract(extract_timer_str.c_str(),
                                                    RNAME);
    std::string extract_batch_timer_str(syncTypeStr + "ExtractBatch_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Textractbatch(
        extract_batch_timer_str.c_str(), RNAME);

    DataCommMode data_mode;

    Textract.start();

    if (num > 0) {
      data_mode = onlyData;
      b.reserve(sizeof(DataCommMode) + sizeof(size_t) +
                (num * sizeof(typename SyncFnTy::ValTy)));

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
            b, num,
            (galois::PODResizeableArray<typename SyncFnTy::ValTy>*)nullptr);
        extractSubset<SyncFnTy, decltype(lseq), syncType, true, true>(
            loopName, indices, num, offsets, b, lseq);
      } else {
        b.resize(sizeof(DataCommMode) + sizeof(size_t) +
                 (num * sizeof(typename SyncFnTy::ValTy)));
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
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(RNAME, metadata_str,
                                                            1);
  }

  /**
   * Non-bitset extract for when the type of the item being sync'd isn't
   * memory copyable.
   *
   * Extracts all of the data for all edges in indices and saves it into
   * a send buffer for return.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam syncFnTy struct that has info on how to do synchronization
   *
   * @param loopName loop name used for timers
   * @param from_id
   * @param indices Vector that contains node ids of edges that we will
   * potentially send things to
   * @param b OUTPUT: buffer that will be sent over the network; contains data
   * based on set bits in bitset
   */
  template <SyncType syncType, typename SyncFnTy, bool async,
            typename std::enable_if<!galois::runtime::is_memory_copyable<
                typename SyncFnTy::ValTy>::value>::type* = nullptr>
  void syncExtract(std::string loopName, unsigned from_id,
                   std::vector<size_t>& indices,
                   galois::runtime::SendBuffer& b) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string extract_timer_str(syncTypeStr + "Extract_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Textract(extract_timer_str.c_str(),
                                                    RNAME);
    std::string extract_batch_timer_str(syncTypeStr + "ExtractBatch_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Textractbatch(
        extract_batch_timer_str.c_str(), RNAME);

    DataCommMode data_mode;

    uint32_t num = indices.size();
    static galois::PODResizeableArray<typename SyncFnTy::ValTy> val_vec;
    static galois::PODResizeableArray<unsigned int> dummyVector;

    Textract.start();

    if (num > 0) {
      data_mode = onlyData;
      b.reserve(sizeof(DataCommMode) + sizeof(size_t) +
                (num * sizeof(typename SyncFnTy::ValTy)));

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
        extractSubset<SyncFnTy, syncType, true, true>(loopName, indices, num,
                                                      dummyVector, val_vec);
        gSerialize(b, onlyData, val_vec);
      } else {
        b.resize(sizeof(DataCommMode) + sizeof(size_t) +
                 (num * sizeof(typename SyncFnTy::ValTy)));
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
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(RNAME, metadata_str,
                                                            1);
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
   * @param indices Vector that contains node ids of edges that we will
   * potentially send things to
   * @param b OUTPUT: buffer that will be sent over the network; contains data
   * based on set bits in bitset
   */
  template <
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, bool async,
      typename std::enable_if<!BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  void syncExtract(std::string loopName, unsigned from_id,
                   std::vector<size_t>& indices,
                   galois::runtime::SendBuffer& b) {
    const galois::DynamicBitSet& bit_set_compute = BitsetFnTy::get();
    uint64_t manualBitsetCount                   = bit_set_compute.count();
    uint32_t num                                 = indices.size();
    galois::DynamicBitSet& bit_set_comm          = syncBitset;
    static galois::PODResizeableArray<typename SyncFnTy::ValTy> val_vec;
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;

    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string extract_timer_str(syncTypeStr + "Extract_" +
                                  get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Textract(extract_timer_str.c_str(),
                                                    RNAME);
    std::string extract_alloc_timer_str(syncTypeStr + "ExtractAlloc_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Textractalloc(
        extract_alloc_timer_str.c_str(), RNAME);
    std::string extract_batch_timer_str(syncTypeStr + "ExtractBatch_" +
                                        get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Textractbatch(
        extract_batch_timer_str.c_str(), RNAME);

    DataCommMode data_mode;

    Textract.start();

    if (num > 0 && manualBitsetCount > 0) {
      // if (num > 0) {
      size_t bit_set_count = 0;
      Textractalloc.start();
      if (substrateDataMode == gidsData) {
        b.reserve(sizeof(DataCommMode) + sizeof(bit_set_count) +
                  sizeof(size_t) + (num * sizeof(unsigned int)) +
                  sizeof(size_t) + (num * sizeof(typename SyncFnTy::ValTy)));
      } else if (substrateDataMode == offsetsData) {
        b.reserve(sizeof(DataCommMode) + sizeof(bit_set_count) +
                  sizeof(size_t) + (num * sizeof(unsigned int)) +
                  sizeof(size_t) + (num * sizeof(typename SyncFnTy::ValTy)));
      } else if (substrateDataMode == bitsetData) {
        size_t bitset_alloc_size = ((num + 63) / 64) * sizeof(uint64_t);
        b.reserve(sizeof(DataCommMode) + sizeof(bit_set_count) +
                  sizeof(size_t)   // bitset size
                  + sizeof(size_t) // bitset vector size
                  + bitset_alloc_size + sizeof(size_t) +
                  (num * sizeof(typename SyncFnTy::ValTy)));
      } else { // onlyData or noData (auto)
        size_t bitset_alloc_size = ((num + 63) / 64) * sizeof(uint64_t);
        b.reserve(sizeof(DataCommMode) + sizeof(bit_set_count) +
                  sizeof(size_t)   // bitset size
                  + sizeof(size_t) // bitset vector size
                  + bitset_alloc_size + sizeof(size_t) +
                  (num * sizeof(typename SyncFnTy::ValTy)));
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

        getBitsetAndOffsets<SyncFnTy, syncType>(
            loopName, indices, bit_set_compute, bit_set_comm, offsets,
            bit_set_count, data_mode);

        if (data_mode == onlyData) {
          bit_set_count = indices.size();
          extractSubset<SyncFnTy, syncType, true, true>(
              loopName, indices, bit_set_count, offsets, val_vec);
        } else if (data_mode !=
                   noData) { // bitsetData or offsetsData or gidsData
          extractSubset<SyncFnTy, syncType, false, true>(
              loopName, indices, bit_set_count, offsets, val_vec);
        }
        serializeMessage<async, syncType>(loopName, data_mode, bit_set_count,
                                          indices, offsets, bit_set_comm,
                                          val_vec, b);
      } else {
        if (data_mode == noData) {
          b.resize(0);
          if (!async) {
            gSerialize(b, data_mode);
          }
        } else if (data_mode == gidsData) {
          b.resize(sizeof(DataCommMode) + sizeof(bit_set_count) +
                   sizeof(size_t) + (bit_set_count * sizeof(unsigned int)) +
                   sizeof(size_t) +
                   (bit_set_count * sizeof(typename SyncFnTy::ValTy)));
        } else if (data_mode == offsetsData) {
          b.resize(sizeof(DataCommMode) + sizeof(bit_set_count) +
                   sizeof(size_t) + (bit_set_count * sizeof(unsigned int)) +
                   sizeof(size_t) +
                   (bit_set_count * sizeof(typename SyncFnTy::ValTy)));
        } else if (data_mode == bitsetData) {
          size_t bitset_alloc_size = ((num + 63) / 64) * sizeof(uint64_t);
          b.resize(sizeof(DataCommMode) + sizeof(bit_set_count) +
                   sizeof(size_t)   // bitset size
                   + sizeof(size_t) // bitset vector size
                   + bitset_alloc_size + sizeof(size_t) +
                   (bit_set_count * sizeof(typename SyncFnTy::ValTy)));
        } else { // onlyData
          b.resize(sizeof(DataCommMode) + sizeof(size_t) +
                   (num * sizeof(typename SyncFnTy::ValTy)));
        }
      }

      reportRedundantSize<SyncFnTy>(loopName, syncTypeStr, num, bit_set_count,
                                    bit_set_comm);
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
    galois::runtime::reportStatCond_Single<MORE_DIST_STATS>(RNAME, metadata_str,
                                                            1);
  }

#ifdef GALOIS_USE_BARE_MPI
  /**
   * Sync using MPI instead of network layer.
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_send(std::string loopName) {
    static std::vector<galois::runtime::SendBuffer> b;
    static std::vector<MPI_Request> request;
    b.resize(numHosts);
    request.resize(numHosts, MPI_REQUEST_NULL);

    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType))
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

    if (BitsetFnTy::is_valid() && syncType == syncBroadcast) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }
  }

  /**
   * Sync put using MPI instead of network layer
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_put(std::string loopName, const MPI_Group& mpi_access_group,
                    const std::vector<MPI_Win>& window) {

    MPI_Win_start(mpi_access_group, 0, window[id]);

    std::vector<galois::runtime::SendBuffer> b(numHosts);
    std::vector<size_t> size(numHosts);
    uint64_t send_buffers_size = 0;

    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType))
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

    if (BitsetFnTy::is_valid() && syncType == syncBroadcast) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }
  }
#endif

  /**
   * Sends data to all hosts (if there is anything that needs to be sent
   * to that particular host) and adjusts bitset according to sync type.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has information needed to access bitset
   *
   * @param loopName used to name timers created by this sync send
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy,
            bool async>
  void syncNetSend(std::string loopName) {
    static galois::runtime::SendBuffer
        b; // although a static variable, allocation not reused
           // due to std::move in net.sendTagged()

    auto& net               = galois::runtime::getSystemNetworkInterface();
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string statNumMessages_str(syncTypeStr + "NumMessages_" +
                                    get_run_identifier(loopName));

    size_t numMessages = 0;
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType))
        continue;

      getSendBuffer<syncType, SyncFnTy, BitsetFnTy, async>(loopName, x, b);

      if ((!async) || (b.size() > 0)) {
        size_t syncTypePhase = 0;
        if (async && (syncType == syncBroadcast))
          syncTypePhase = 1;
        net.sendTagged(x, galois::runtime::evilPhase, b, syncTypePhase);
        ++numMessages;
      }
    }
    if (!async) {
      // Will force all messages to be processed before continuing
      net.flush();
    }

    if (BitsetFnTy::is_valid() && syncType == syncBroadcast) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }

    galois::runtime::reportStat_Tsum(RNAME, statNumMessages_str, numMessages);
  }

  /**
   * Sends data over the network to other hosts based on the provided template
   * arguments.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy,
            bool async>
  void syncSend(std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<GALOIS_COMM_STATS> TSendTime(
        (syncTypeStr + "Send_" + get_run_identifier(loopName)).c_str(), RNAME);

    TSendTime.start();
    syncNetSend<syncType, SyncFnTy, BitsetFnTy, async>(loopName);
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
      SyncType syncType, typename SyncFnTy, typename BitsetFnTy, bool async,
      typename std::enable_if<!BitsetFnTy::is_vector_bitset()>::type* = nullptr>
  size_t syncRecvApply(uint32_t from_id, galois::runtime::RecvBuffer& buf,
                       std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    std::string set_timer_str(syncTypeStr + "Set_" +
                              get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Tset(set_timer_str.c_str(), RNAME);
    std::string set_batch_timer_str(syncTypeStr + "SetBatch_" +
                                    get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Tsetbatch(
        set_batch_timer_str.c_str(), RNAME);

    galois::DynamicBitSet& bit_set_comm = syncBitset;
    static galois::PODResizeableArray<typename SyncFnTy::ValTy> val_vec;
    galois::PODResizeableArray<unsigned int>& offsets = syncOffsets;

    auto& sharedEdges = (syncType == syncReduce) ? masterEdges : mirrorEdges;
    uint32_t num      = sharedEdges[from_id].size();
    size_t retval     = 0;
    Tset.start();

    if (num > 0) { // only enter if we expect message from that host
      DataCommMode data_mode;
      // 1st deserialize gets data mode
      galois::runtime::gDeserialize(buf, data_mode);

      if (data_mode != noData) {
        // GPU update call
        Tsetbatch.start();
        bool batch_succeeded =
            setBatchWrapper<SyncFnTy, syncType, async>(from_id, buf, data_mode);
        Tsetbatch.stop();

        // cpu always enters this block
        if (!batch_succeeded) {
          size_t bit_set_count = num;
          size_t buf_start     = 0;

          // deserialize the rest of the data in the buffer depending on the
          // data mode; arguments passed in here are mostly output vars
          deserializeMessage<syncType>(loopName, data_mode, num, buf,
                                       bit_set_count, offsets, bit_set_comm,
                                       buf_start, retval, val_vec);

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
            setSubset<decltype(sharedEdges[from_id]), SyncFnTy, syncType, async,
                      true, true>(loopName, sharedEdges[from_id], bit_set_count,
                                  offsets, val_vec, bit_set_compute);
          } else if (data_mode == dataSplit || data_mode == dataSplitFirst) {
            setSubset<decltype(sharedEdges[from_id]), SyncFnTy, syncType, async,
                      true, true>(loopName, sharedEdges[from_id], bit_set_count,
                                  offsets, val_vec, bit_set_compute, buf_start);
          } else if (data_mode == gidsData) {
            setSubset<decltype(offsets), SyncFnTy, syncType, async, true, true>(
                loopName, offsets, bit_set_count, offsets, val_vec,
                bit_set_compute);
          } else { // bitsetData or offsetsData
            setSubset<decltype(sharedEdges[from_id]), SyncFnTy, syncType, async,
                      false, true>(loopName, sharedEdges[from_id],
                                   bit_set_count, offsets, val_vec,
                                   bit_set_compute);
          }
          // TODO: reduce could update the bitset, so it needs to be copied
          // back to the device
        }
      }
    }

    Tset.stop();

    return retval;
  }

#ifdef GALOIS_USE_BARE_MPI
  /**
   * MPI Irecv wrapper for sync
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_recv_post(std::string loopName,
                          std::vector<MPI_Request>& request,
                          const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType))
        continue;

      MPI_Irecv((uint8_t*)rb[x].data(), rb[x].size(), MPI_BYTE, x, 32767,
                MPI_COMM_WORLD, &request[x]);
    }
  }

  /**
   * MPI receive wrapper for sync
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_recv_wait(std::string loopName,
                          std::vector<MPI_Request>& request,
                          const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType))
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
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_get(std::string loopName, const std::vector<MPI_Win>& window,
                    const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType))
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
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy,
            bool async>
  void syncNetRecv(std::string loopName) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string wait_timer_str("Wait_" + get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> Twait(wait_timer_str.c_str(), RNAME);

    if (async) {
      size_t syncTypePhase = 0;
      if (syncType == syncBroadcast)
        syncTypePhase = 1;
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr,
                                 syncTypePhase)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr,
                              syncTypePhase);

        if (p) {
          syncRecvApply<syncType, SyncFnTy, BitsetFnTy, async>(
              p->first, p->second, loopName);
        }
      } while (p);
    } else {
      for (unsigned x = 0; x < numHosts; ++x) {
        if (x == id)
          continue;
        if (nothingToRecv(x, syncType))
          continue;

        Twait.start();
        decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
        do {
          p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        } while (!p);
        Twait.stop();

        syncRecvApply<syncType, SyncFnTy, BitsetFnTy, async>(
            p->first, p->second, loopName);
      }
      incrementEvilPhase();
    }
  }

  /**
   * Receives messages from all other hosts and "applies" the message (reduce
   * or set) based on the sync structure provided.
   *
   * @tparam syncType either reduce or broadcast
   * @tparam SyncFnTy synchronization structure with info needed to synchronize
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy,
            bool async>
  void syncRecv(std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<GALOIS_COMM_STATS> TRecvTime(
        (syncTypeStr + "Recv_" + get_run_identifier(loopName)).c_str(), RNAME);

    TRecvTime.start();
    syncNetRecv<syncType, SyncFnTy, BitsetFnTy, async>(loopName);
    TRecvTime.stop();
  }

////////////////////////////////////////////////////////////////////////////////
// MPI sync variants
////////////////////////////////////////////////////////////////////////////////
#ifdef GALOIS_USE_BARE_MPI
  /**
   * Nonblocking MPI sync
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void syncNonblockingMPI(std::string loopName,
                          bool use_bitset_to_send = true) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<GALOIS_COMM_STATS> TSendTime(
        (syncTypeStr + "Send_" + get_run_identifier(loopName)).c_str(), RNAME);
    galois::CondStatTimer<GALOIS_COMM_STATS> TRecvTime(
        (syncTypeStr + "Recv_" + get_run_identifier(loopName)).c_str(), RNAME);

    static std::vector<std::vector<uint8_t>> rb;
    static std::vector<MPI_Request> request;

    if (rb.size() == 0) { // create the receive buffers
      TRecvTime.start();
      auto& sharedEdges = (syncType == syncReduce) ? masterEdges : mirrorEdges;
      rb.resize(numHosts);
      request.resize(numHosts, MPI_REQUEST_NULL);

      for (unsigned h = 1; h < numHosts; ++h) {
        unsigned x = (id + numHosts - h) % numHosts;
        if (nothingToRecv(x, syncType))
          continue;

        size_t size =
            (sharedEdges[x].size() * sizeof(typename SyncFnTy::ValTy));
        size += sizeof(size_t);       // vector size
        size += sizeof(DataCommMode); // data mode

        rb[x].resize(size);
      }
      TRecvTime.stop();
    }

    TRecvTime.start();
    sync_mpi_recv_post<syncType, SyncFnTy, BitsetFnTy>(loopName, request, rb);
    TRecvTime.stop();

    TSendTime.start();
    if (use_bitset_to_send) {
      sync_mpi_send<syncType, SyncFnTy, BitsetFnTy>(loopName);
    } else {
      sync_mpi_send<syncType, SyncFnTy, galois::InvalidBitsetFnTy>(loopName);
    }
    TSendTime.stop();

    TRecvTime.start();
    sync_mpi_recv_wait<syncType, SyncFnTy, BitsetFnTy>(loopName, request, rb);
    TRecvTime.stop();
  }

  /**
   * Onesided MPI sync
   */
  template <SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void syncOnesidedMPI(std::string loopName, bool use_bitset_to_send = true) {
    std::string syncTypeStr = (syncType == syncReduce) ? "Reduce" : "Broadcast";
    galois::CondStatTimer<GALOIS_COMM_STATS> TSendTime(
        (syncTypeStr + "Send_" + get_run_identifier(loopName)).c_str(), RNAME);
    galois::CondStatTimer<GALOIS_COMM_STATS> TRecvTime(
        (syncTypeStr + "Recv_" + get_run_identifier(loopName)).c_str(), RNAME);

    static std::vector<MPI_Win> window;
    static MPI_Group mpi_access_group;
    static std::vector<std::vector<uint8_t>> rb;

    if (window.size() == 0) { // create the windows
      TRecvTime.start();
      auto& sharedEdges = (syncType == syncReduce) ? masterEdges : mirrorEdges;
      window.resize(numHosts);
      rb.resize(numHosts);

      uint64_t recv_buffers_size = 0;
      for (unsigned x = 0; x < numHosts; ++x) {
        size_t size = sharedEdges[x].size() * sizeof(typename SyncFnTy::ValTy);
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
        if (nothingToRecv(x, syncType))
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

        if (nothingToSend(x, syncType))
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
      sync_mpi_put<syncType, SyncFnTy, BitsetFnTy>(loopName, mpi_access_group,
                                                   window);
    } else {
      sync_mpi_put<syncType, SyncFnTy, galois::InvalidBitsetFnTy>(
          loopName, mpi_access_group, window);
    }
    TSendTime.stop();

    TRecvTime.start();
    sync_mpi_get<syncType, SyncFnTy, BitsetFnTy>(loopName, window, rb);
    TRecvTime.stop();
  }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  // Higher Level Sync Calls (broadcast/reduce, etc)
  ////////////////////////////////////////////////////////////////////////////////

  /**
   * Does a reduction of data from mirror edges to master edges.
   *
   * @tparam ReduceFnTy reduce sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename ReduceFnTy, typename BitsetFnTy, bool async>
  inline void reduce(std::string loopName) {
    std::string timer_str("Reduce_" + get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> TsyncReduce(timer_str.c_str(),
                                                       RNAME);
    TsyncReduce.start();

#ifdef GALOIS_USE_BARE_MPI
    switch (bare_mpi) {
    case noBareMPI:
#endif
      syncSend<syncReduce, ReduceFnTy, BitsetFnTy, async>(loopName);
      syncRecv<syncReduce, ReduceFnTy, BitsetFnTy, async>(loopName);
#ifdef GALOIS_USE_BARE_MPI
      break;
    case nonBlockingBareMPI:
      syncNonblockingMPI<syncReduce, ReduceFnTy, BitsetFnTy>(loopName);
      break;
    case oneSidedBareMPI:
      syncOnesidedMPI<syncReduce, ReduceFnTy, BitsetFnTy>(loopName);
      break;
    default:
      GALOIS_DIE("Unsupported bare MPI");
    }
#endif

    TsyncReduce.stop();
  }

  /**
   * Does a broadcast of data from master to mirror edges.
   *
   * @tparam BroadcastFnTy broadcast sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   *
   * @param loopName used to name timers for statistics
   */
  template <typename BroadcastFnTy, typename BitsetFnTy, bool async>
  inline void broadcast(std::string loopName) {
    std::string timer_str("Broadcast_" + get_run_identifier(loopName));
    galois::CondStatTimer<GALOIS_COMM_STATS> TsyncBroadcast(timer_str.c_str(),
                                                          RNAME);

    TsyncBroadcast.start();

    bool use_bitset = true;

#ifdef GALOIS_USE_BARE_MPI
    switch (bare_mpi) {
    case noBareMPI:
#endif
      if (use_bitset) {
        syncSend<syncBroadcast, BroadcastFnTy, BitsetFnTy, async>(loopName);
      } else {
        syncSend<syncBroadcast, BroadcastFnTy, galois::InvalidBitsetFnTy,
                 async>(loopName);
      }
      syncRecv<syncBroadcast, BroadcastFnTy, BitsetFnTy, async>(loopName);
#ifdef GALOIS_USE_BARE_MPI
      break;
    case nonBlockingBareMPI:
      syncNonblockingMPI<syncBroadcast, BroadcastFnTy, BitsetFnTy>(loopName,
                                                                   use_bitset);
      break;
    case oneSidedBareMPI:
      syncOnesidedMPI<syncBroadcast, BroadcastFnTy, BitsetFnTy>(loopName,
                                                                use_bitset);
      break;
    default:
      GALOIS_DIE("Unsupported bare MPI");
    }
#endif

    TsyncBroadcast.stop();
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
    reduce<SyncFnTy, BitsetFnTy, async>(loopName);
    broadcast<SyncFnTy, BitsetFnTy, async>(loopName);
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
   * @tparam SyncFnTy sync structure for the field
   * @tparam BitsetFnTy struct that has info on how to access the bitset
   * @param loopName used to name timers for statistics
   */
  template <typename SyncFnTy, typename BitsetFnTy = galois::InvalidBitsetFnTy,
            bool async = false>
  inline void sync(std::string loopName) {
    std::string timer_str("Sync_" + loopName + "_" + get_run_identifier());
    galois::StatTimer Tsync(timer_str.c_str(), RNAME);

    Tsync.start();
    sync_any_to_any<SyncFnTy, BitsetFnTy, async>(loopName);
    Tsync.stop();
  }

  ////////////////////////////////////////////////////////////////////////////////
  // GPU marshaling
  ////////////////////////////////////////////////////////////////////////////////

#ifdef GALOIS_ENABLE_GPU
private:
  using GraphNode     = typename GraphTy::GraphNode;
  using edge_iterator = typename GraphTy::edge_iterator;
  using EdgeTy        = typename GraphTy::EdgeType;

  // Code that handles getting the graph onto the GPU
  template <bool isVoidType,
            typename std::enable_if<isVoidType>::type* = nullptr>
  inline void setMarshalEdge(EdgeMarshalGraph& m, const size_t index,
                             const edge_iterator& e) {
    // do nothing
  }

  template <bool isVoidType,
            typename std::enable_if<!isVoidType>::type* = nullptr>
  inline void setMarshalEdge(EdgeMarshalGraph& m, const size_t index,
                             const edge_iterator& e) {
    m.edge_data[index] = userGraph.getEdgeData(e);
  }

public:
  void getEdgeMarshalGraph(EdgeMarshalGraph& m, bool loadProxyEdges = true) {
    m.nnodes   = userGraph.size();
    m.nedges   = userGraph.sizeEdges();
    m.numOwned = userGraph.numMasters();
    //// Assumption: master occurs at beginning in contiguous range
    m.beginMaster       = 0;
    m.numNodesWithEdges = userGraph.getNumNodesWithEdges();
    m.id                = id;
    m.numHosts          = numHosts;
    m.row_start         = (index_type*)calloc(m.nnodes + 1, sizeof(index_type));
    m.edge_dst          = (index_type*)calloc(m.nedges, sizeof(index_type));
    m.node_data         = (index_type*)calloc(m.nnodes, sizeof(node_data_type));

    //// TODO deal with edgety
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
          m.node_data[nodeID] =
              userGraph.getGID(nodeID); // this may not be required.
          m.row_start[nodeID] = *(userGraph.edge_begin(nodeID));
          for (auto e = userGraph.edge_begin(nodeID);
               e != userGraph.edge_end(nodeID); e++) {
            auto edgeID = *e;
            setMarshalEdge<std::is_void<EdgeTy>::value>(m, edgeID, e);
            m.edge_dst[edgeID] = userGraph.getEdgeDst(e);
          }
        },
        galois::steal());

    m.row_start[m.nnodes] = m.nedges;

    // TODO?
    // copy memoization meta-data
    if (loadProxyEdges) {
      m.num_master_edges =
          (unsigned int*)calloc(masterEdges.size(), sizeof(unsigned int));
      ;
      m.master_edges =
          (unsigned int**)calloc(masterEdges.size(), sizeof(unsigned int*));
      ;

      for (uint32_t h = 0; h < masterEdges.size(); ++h) {
        m.num_master_edges[h] = masterEdges[h].size();

        if (masterEdges[h].size() > 0) {
          m.master_edges[h] = (unsigned int*)calloc(masterEdges[h].size(),
                                                    sizeof(unsigned int));
          ;
          std::copy(masterEdges[h].begin(), masterEdges[h].end(),
                    m.master_edges[h]);
        } else {
          m.master_edges[h] = NULL;
        }
      }

      m.num_mirror_edges =
          (unsigned int*)calloc(mirrorEdges.size(), sizeof(unsigned int));
      ;
      m.mirror_edges =
          (unsigned int**)calloc(mirrorEdges.size(), sizeof(unsigned int*));
      ;
      for (uint32_t h = 0; h < mirrorEdges.size(); ++h) {
        m.num_mirror_edges[h] = mirrorEdges[h].size();

        if (mirrorEdges[h].size() > 0) {
          m.mirror_edges[h] = (unsigned int*)calloc(mirrorEdges[h].size(),
                                                    sizeof(unsigned int));
          ;
          std::copy(mirrorEdges[h].begin(), mirrorEdges[h].end(),
                    m.mirror_edges[h]);
        } else {
          m.mirror_edges[h] = NULL;
        }
      }
    }

    //// user needs to provide method of freeing up graph (it can do nothing
    //// if they wish)
    // userGraph.deallocate();
  }
#endif // het galois def

public:
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
#if GALOIS_PER_ROUND_STATS
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
#if GALOIS_PER_ROUND_STATS
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
#if GALOIS_PER_ROUND_STATS
    return std::string(std::string(loop_name) + "_" + std::to_string(alterID) +
                       "_" + std::to_string(num_run) + "_" +
                       std::to_string(num_round));
#else
    return std::string(std::string(loop_name) + "_" + std::to_string(alterID) +
                       "_" + std::to_string(num_run));
#endif
  }
};

template <typename GraphTy>
constexpr const char* const galois::graphs::GluonEdgeSubstrate<GraphTy>::RNAME;
} // end namespace graphs
} // end namespace galois

#endif // header guard
