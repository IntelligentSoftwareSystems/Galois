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
 * @file DReducible.h
 *
 * Implements distributed reducible objects for easy reduction of values
 * across a distributed system.
 */
#ifndef GALOIS_DISTTERMINATOR_H
#define GALOIS_DISTTERMINATOR_H

#include <limits>
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/AtomicHelpers.h"
#include "galois/runtime/LWCI.h"
#include "galois/runtime/DistStats.h"

namespace galois {

/**
 * Distributed sum-reducer for getting the sum of some value across multiple
 * hosts.
 *
 * @tparam Ty type of value to max-reduce
 */
template <typename Ty>
class DGTerminator {
  galois::runtime::NetworkInterface& net =
      galois::runtime::getSystemNetworkInterface();

  galois::GAccumulator<Ty> mdata;
  Ty local_mdata, global_mdata;

  uint64_t prev_snapshot;
  uint64_t snapshot;
  uint64_t global_snapshot;
  bool work_done;
#ifndef GALOIS_USE_LWCI
  MPI_Request snapshot_request;
#endif

public:
  //! Default constructor
  DGTerminator() {
    reinitialize();
    initiate_snapshot();
  }

  void reinitialize() {
    prev_snapshot = 0;
    snapshot = 1; 
    global_snapshot = 1;
    work_done = false;
  }

  /**
   * Adds to accumulated value
   *
   * @param rhs Value to add
   * @returns reference to this object
   */
  DGTerminator& operator+=(const Ty& rhs) {
    mdata += rhs;
    return *this;
  }

  /**
   * Sets current value stored in accumulator.
   *
   * @param rhs Value to set
   */
  void operator=(const Ty rhs) {
    mdata.reset();
    mdata += rhs;
  }

  /**
   * Sets current value stored in accumulator.
   *
   * @param rhs Value to set
   */
  void set(const Ty rhs) {
    mdata.reset();
    mdata += rhs;
  }

  /**
   * Read local accumulated value.
   *
   * @returns locally accumulated value
   */
  Ty read_local() {
    if (local_mdata == 0)
      local_mdata = mdata.reduce();
    return local_mdata;
  }

  /**
   * Read the value returned by the last reduce call.
   * Should call reduce before calling this function if an up to date
   * value is required
   *
   * @returns the value of the last reduce call
   */
  Ty read() { return global_mdata; }

  /**
   * Reset the entire accumulator.
   *
   * @returns the value of the last reduce call
   */
  Ty reset() {
    Ty retval = global_mdata;
    mdata.reset();
    local_mdata = global_mdata = 0;
    return retval;
  }

  void initiate_snapshot() {
#ifdef GALOIS_USE_LWCI
    assert(false);
    lc_alreduce(&snapshot, &global_snapshot, sizeof(Ty),
                &galois::runtime::internal::ompi_op_max<Ty>, &snapshot_request);
#else
    MPI_Iallreduce(&snapshot, &global_snapshot, 1, MPI::UNSIGNED_LONG, MPI_MAX,
                  MPI_COMM_WORLD, &snapshot_request);
#endif
  }

  bool terminate() {
    bool active = (local_mdata != 0);
    if (active) galois::gDebug("[", net.ID, "] local work done \n");
    if (!active) {
      active = net.anyPendingSends();
      if (active) galois::gDebug("[", net.ID, "] pending send \n");
      if (!active) {
        active = net.anyPendingReceives();
        if (active) galois::gDebug("[", net.ID, "] pending receive \n");
      }
    }
    if (active) {
      work_done = true;
    } else {
      int snapshot_ended = 0;
      MPI_Test(&snapshot_request, &snapshot_ended, MPI_STATUS_IGNORE);
      if (snapshot_ended != 0) {
        snapshot = global_snapshot;
        if (work_done) {
          work_done = false;
          prev_snapshot = snapshot;
          ++snapshot;
          galois::gDebug("[", net.ID, "] work done, taking snapshot ", snapshot, " \n");
          initiate_snapshot();
        } else if (prev_snapshot != snapshot) {
          prev_snapshot = snapshot;
          galois::gDebug("[", net.ID, "] no work done, taking snapshot ", snapshot, " \n");
          initiate_snapshot();
        } else {
          galois::gDebug("[", net.ID, "] terminating ", snapshot, " \n");
          reinitialize(); // for next async phase
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Reduce data across all hosts, saves the value, and returns the
   * reduced value
   *
   * @param runID optional argument used to create a statistics timer
   * for later reporting
   *
   * @returns The reduced value
   */
  Ty reduce(std::string runID = std::string()) {
    std::string timer_str("ReduceDGAccum_" + runID);

    galois::CondStatTimer<MORE_DIST_STATS> reduceTimer(timer_str.c_str(),
                                                       "DGReducible");
    reduceTimer.start();

    if (local_mdata == 0)
      local_mdata = mdata.reduce();

    bool halt = terminate();
    global_mdata = !halt;
    if (halt) {
      ++galois::runtime::evilPhase;
      if (galois::runtime::evilPhase >=
          std::numeric_limits<int16_t>::max()) { // limit defined by MPI or LCI
        galois::runtime::evilPhase = 1;
      }
    }

    reduceTimer.stop();

    return global_mdata;
  }
};

} // namespace galois
#endif
