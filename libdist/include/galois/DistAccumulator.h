/** Distributed Accumulator type -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#ifndef GALOIS_DISTACCUMULATOR_H
#define GALOIS_DISTACCUMULATOR_H

#include <limits>
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/AtomicHelpers.h"
#include "galois/runtime/LWCI.h"

namespace galois {

template<typename Ty>
class DGAccumulator {
  galois::runtime::NetworkInterface& net = galois::runtime::getSystemNetworkInterface();

  galois::GAccumulator<Ty> mdata;
  Ty local_mdata, global_mdata;

#ifdef GALOIS_USE_LWCI
  inline void reduce_lwci() {
    lc_alreduce(&local_mdata, &global_mdata, sizeof(Ty), &ompi_op_sum<Ty>, mv);
  }
#else
  inline void reduce_mpi() {
    if (typeid(Ty) == typeid(int32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI::INT, MPI_SUM, MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(int64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI::LONG, MPI_SUM, MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint32_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI::UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(uint64_t)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI::UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(float)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI::FLOAT, MPI_SUM, MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } else if (typeid(Ty) == typeid(long double)) {
      MPI_Allreduce(&local_mdata, &global_mdata, 1, MPI::LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } else {
      static_assert(true, "Type of DGAccumulator not supported for MPI reduction");
    }
  }
#endif

public:
  // Default constructor
  DGAccumulator() {}

  DGAccumulator& operator+=(const Ty& rhs) {
    mdata += rhs;
    return *this;
  }

  void operator=(const Ty rhs) {
    mdata.reset();
    mdata += rhs;
  }

  void set(const Ty rhs) {
    mdata.reset();
    mdata += rhs;
  }

  Ty read_local() {
    if (local_mdata == 0) local_mdata = mdata.reduce();
    return local_mdata;
  }

  Ty read() {
    return global_mdata;
  }

  Ty reset() {
    Ty retval = global_mdata;
    mdata.reset();
    local_mdata = global_mdata = 0;
    return retval;
  }

  /************************************************************
   *
   ************************************************************/

  /**
   * Reduce with a timer
   */
  Ty reduce(std::string runID = std::string()) {
    std::string timer_str("REDUCE_DGACCUM_" + runID);
    galois::StatTimer reduceTimer(timer_str.c_str(), "DGReducible");
    reduceTimer.start();

    if (local_mdata == 0) local_mdata = mdata.reduce();

#ifdef GALOIS_USE_LWCI 
    reduce_lwci();
#else
    reduce_mpi();
#endif

    reduceTimer.stop();

    return global_mdata;
  }


  /************************************************************
   *
   ************************************************************/

  /* Max reduction across DGAccumulators */
  // TODO: FIX THIS - should be a separate reduction type, not within Accumulator
  Ty reduce_max() {
    if (local_mdata == 0) local_mdata = mdata.reduce();
    global_mdata = local_mdata;
    for (unsigned h = 1; h < net.Num; ++h) {
      unsigned x = (net.ID + h) % net.Num;
      galois::runtime::SendBuffer b;
      gSerialize(b, local_mdata);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }
    net.flush();

    unsigned num_Hosts_recvd = 1;
    while (num_Hosts_recvd < net.Num) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      Ty x_mdata;
      gDeserialize(p->second, x_mdata);
      galois::max(global_mdata, x_mdata);
      ++num_Hosts_recvd;
    }
    ++galois::runtime::evilPhase;

    // returns max from all accumulators
    return global_mdata;
  }

  /************************************************************
   *
   ************************************************************/

  /* Min reduction across DGAccumulators */
  // TODO: FIX THIS - should be a separate reduction type, not within Accumulator
  Ty reduce_min() {
    if (local_mdata == 0) local_mdata = mdata.reduce();
    global_mdata = local_mdata;
    for (unsigned h = 1; h < net.Num; ++h) {
      unsigned x = (net.ID + h) % net.Num;
      galois::runtime::SendBuffer b;
      gSerialize(b, local_mdata);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }
    net.flush();

    unsigned num_Hosts_recvd = 1;
    while (num_Hosts_recvd < net.Num) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      Ty x_mdata;
      gDeserialize(p->second, x_mdata);
      galois::min(global_mdata, x_mdata);
      ++num_Hosts_recvd;
    }
    ++galois::runtime::evilPhase;

    // returns min from all accumulators
    return global_mdata;
  }
};
}
#endif
