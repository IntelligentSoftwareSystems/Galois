/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
#ifndef GALOIS_DISTACCUMULATOR_H
#define GALOIS_DISTACCUMULATOR_H

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
class DGAccumulator {
  galois::runtime::NetworkInterface& net =
      galois::runtime::getSystemNetworkInterface();

  galois::GAccumulator<Ty> mdata;
  Ty local_mdata, global_mdata;

#ifdef GALOIS_USE_LCI
  /**
   * Sum reduction using LWCI
   */
  inline void reduce_lwci() {
    lc_alreduce(&local_mdata, &global_mdata, sizeof(Ty),
                &galois::runtime::internal::ompi_op_sum<Ty>, lc_col_ep);
  }
#else
  /**
   * Sum reduction using MPI
   */
  inline void reduce_mpi() {
    if (typeid(Ty) == typeid(int32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(int64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_LONG, MPI_SUM,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_UNSIGNED, MPI_SUM,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(float)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(long double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_LONG_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    } else {
      static_assert(true,
                    "Type of DGAccumulator not supported for MPI reduction");
    }
  }
#endif

public:
  //! Default constructor
  DGAccumulator() {}

  /**
   * Adds to accumulated value
   *
   * @param rhs Value to add
   * @returns reference to this object
   */
  DGAccumulator& operator+=(const Ty& rhs) {
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

    galois::CondStatTimer<GALOIS_COMM_STATS> reduceTimer(timer_str.c_str(),
                                                         "DGReducible");
    reduceTimer.start();

    if (local_mdata == 0)
      local_mdata = mdata.reduce();

#ifdef GALOIS_USE_LCI
    reduce_lwci();
#else
    reduce_mpi();
#endif

    reduceTimer.stop();

    return global_mdata;
  }
};

////////////////////////////////////////////////////////////////////////////////

/**
 * Distributed max-reducer for getting the max of some value across multiple
 * hosts.
 *
 * @tparam Ty type of value to max-reduce
 */
template <typename Ty>
class DGReduceMax {
  galois::runtime::NetworkInterface& net =
      galois::runtime::getSystemNetworkInterface();

  galois::GReduceMax<Ty> mdata; // local max reducer
  Ty local_mdata, global_mdata;

#ifdef GALOIS_USE_LCI
  /**
   * Use LWCI to reduce max across hosts
   */
  inline void reduce_lwci() {
    lc_alreduce(&local_mdata, &global_mdata, sizeof(Ty),
                &galois::runtime::internal::ompi_op_max<Ty>, lc_col_ep);
  }
#else
  /**
   * Use MPI to reduce max across hosts
   */
  inline void reduce_mpi() {
    if (typeid(Ty) == typeid(int32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(int64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_LONG, MPI_MAX,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_UNSIGNED, MPI_MAX,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(float)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_FLOAT, MPI_MAX,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(long double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_LONG_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
    } else {
      static_assert(true, "Type of DGReduceMax not supported for MPI "
                          "reduction");
    }
  }
#endif

public:
  /**
   * Default constructor; initializes everything to 0.
   */
  DGReduceMax() {
    local_mdata  = 0;
    global_mdata = 0;
  }

  /**
   * Update the local max-reduced value.
   *
   * @param rhs Value to max-reduce locally with
   */
  void update(const Ty rhs) { mdata.update(rhs); }

  /**
   * Read the local reduced max value; if it has never been reduced, it will
   * attempt get the global value through a reduce (i.e. all other hosts
   * should call reduce as well).
   *
   * @returns the local value stored in the accumulator or a global value if
   * reduce has never been called
   */
  Ty read_local() {
    if (local_mdata == 0)
      local_mdata = mdata.reduce();
    return local_mdata;
  }

  /**
   * Read the global reduced max value. For accurate results, you should
   * call reduce before calling this.
   *
   * @returns the global value stored in the accumulator
   */
  Ty read() { return global_mdata; }

  /**
   * Reset this accumulator.
   *
   * @returns the previous global value stored in this accumulator (note if
   * never reduced, it will be 0
   */
  Ty reset() {
    Ty retval = global_mdata;
    mdata.reset();
    local_mdata = global_mdata = 0;
    return retval;
  }

  /**
   * Do a max reduction across all hosts by sending data to all other hosts
   * and reducing received data.
   *
   * @returns the max-reduced value after reducing from all hosts.
   */
  Ty reduce(std::string runID = std::string()) {
    std::string timer_str("ReduceDGReduceMax_" + runID);

    galois::CondStatTimer<GALOIS_COMM_STATS> reduceTimer(timer_str.c_str(),
                                                         "DGReduceMax");

    reduceTimer.start();
    if (local_mdata == 0)
      local_mdata = mdata.reduce();

#ifdef GALOIS_USE_LCI
    reduce_lwci();
#else
    reduce_mpi();
#endif
    reduceTimer.stop();

    return global_mdata;
  }
};

////////////////////////////////////////////////////////////////////////////////

/**
 * Distributed min-reducer for getting the min of some value across multiple
 * hosts.
 *
 * @tparam Ty type of value to min-reduce
 */
template <typename Ty>
class DGReduceMin {
  galois::runtime::NetworkInterface& net =
      galois::runtime::getSystemNetworkInterface();

  galois::GReduceMin<Ty> mdata; // local min reducer
  Ty local_mdata, global_mdata;

#ifdef GALOIS_USE_LCI
  /**
   * Use LWCI to reduce min across hosts
   */
  inline void reduce_lwci() {
    lc_alreduce(&local_mdata, &global_mdata, sizeof(Ty),
                &galois::runtime::internal::ompi_op_min<Ty>, lc_col_ep);
  }
#else
  /**
   * Use MPI to reduce min across hosts
   */
  inline void reduce_mpi() {
    if (typeid(Ty) == typeid(int32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(int64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_LONG, MPI_MIN,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_UNSIGNED, MPI_MIN,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_UNSIGNED_LONG, MPI_MIN,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(float)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_FLOAT, MPI_MIN,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(long double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI_LONG_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
    } else {
      static_assert(true, "Type of DGReduceMin not supported for MPI "
                          "reduction");
    }
  }
#endif

public:
  /**
   * Default constructor; initializes everything to the max value of the type.
   */
  DGReduceMin() {
    local_mdata  = std::numeric_limits<Ty>::max();
    global_mdata = std::numeric_limits<Ty>::max();
    ;
  }

  /**
   * Update the local min-reduced value.
   *
   * @param rhs Value to min-reduce locally with
   */
  void update(const Ty rhs) { mdata.update(rhs); }

  /**
   * Read the local reduced min value; if it has never been reduced, it will
   * attempt get the global value through a reduce (i.e. all other hosts
   * should call reduce as well).
   *
   * @returns the local value stored in the accumulator or a global value if
   * reduce has never been called
   */
  Ty read_local() {
    if (local_mdata == std::numeric_limits<Ty>::max())
      local_mdata = mdata.reduce();
    return local_mdata;
  }

  /**
   * Read the global reduced min value. For accurate results, you should
   * call reduce before calling this.
   *
   * @returns the global value stored in the accumulator
   */
  Ty read() { return global_mdata; }

  /**
   * Reset this accumulator.
   *
   * @returns the previous global value stored in this accumulator (note if
   * never reduced, it will be 0
   */
  Ty reset() {
    Ty retval = global_mdata;
    mdata.reset();
    local_mdata = global_mdata = std::numeric_limits<Ty>::max();
    return retval;
  }

  /**
   * Do a min reduction across all hosts by sending data to all other hosts
   * and reducing received data.
   *
   * @returns the min-reduced value after reducing from all hosts.
   */
  Ty reduce(std::string runID = std::string()) {
    std::string timer_str("ReduceDGReduceMin_" + runID);

    galois::CondStatTimer<GALOIS_COMM_STATS> reduceTimer(timer_str.c_str(),
                                                         "DGReduceMin");

    reduceTimer.start();
    if (local_mdata == std::numeric_limits<Ty>::max())
      local_mdata = mdata.reduce();

#ifdef GALOIS_USE_LCI
    reduce_lwci();
#else
    reduce_mpi();
#endif
    reduceTimer.stop();

    return global_mdata;
  }
};

} // namespace galois
#endif
