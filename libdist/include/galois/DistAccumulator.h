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
#include "galois/runtime/BareMPI.h"

#ifdef __GALOIS_HET_OPENCL__
#include "galois/opencl/CL_Header.h"
#endif

namespace galois {

template<typename Ty>
class DGAccumulator {
  galois::runtime::NetworkInterface& net = galois::runtime::getSystemNetworkInterface();

  galois::GAccumulator<Ty> mdata;
  Ty local_mdata, global_mdata;
#ifdef __GALOIS_HET_OPENCL__
  cl_mem dev_data;
#endif

  void reduce_net() {
    global_mdata = local_mdata;
#ifdef __GALOIS_HET_OPENCL__
    Ty tmp;
    galois::opencl::CLContext * ctx = galois::opencl::getCLContext();
    cl_int err = clEnqueueReadBuffer(ctx->get_default_device()->command_queue(), dev_data, CL_TRUE, 0, sizeof(Ty), &tmp, 0, NULL, NULL);
//    fprintf(stderr, "READ-DGA[%d, %d]\n", galois::runtime::NetworkInterface::ID, tmp);
    galois::opencl::CHECK_CL_ERROR(err, "Error reading DGAccumulator!\n");
    galois::atomicAdd(mdata, tmp);
#endif
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
      global_mdata += x_mdata;
      ++num_Hosts_recvd;
    }
    ++galois::runtime::evilPhase;
  }

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  void reduce_mpi(const int32_t& l_mdata) {
    MPI_Allreduce(&l_mdata, &global_mdata, 1, MPI::INT, MPI_SUM, MPI_COMM_WORLD);
  }

  void reduce_mpi(const int64_t& l_mdata) {
    MPI_Allreduce(&l_mdata, &global_mdata, 1, MPI::LONG, MPI_SUM, MPI_COMM_WORLD);
  }

  void reduce_mpi(const uint32_t& l_mdata) {
    MPI_Allreduce(&l_mdata, &global_mdata, 1, MPI::UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  }

  void reduce_mpi(const uint64_t& l_mdata) {
    MPI_Allreduce(&l_mdata, &global_mdata, 1, MPI::UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  }

  void reduce_mpi(const float& l_mdata) {
    MPI_Allreduce(&l_mdata, &global_mdata, 1, MPI::FLOAT, MPI_SUM, MPI_COMM_WORLD);
  }

  void reduce_mpi(const double& l_mdata) {
    MPI_Allreduce(&l_mdata, &global_mdata, 1, MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  void reduce_mpi(const long double& l_mdata) {
    MPI_Allreduce(&l_mdata, &global_mdata, 1, MPI::LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif

public:
  // Default constructor
  DGAccumulator() {
#ifdef __GALOIS_HET_OPENCL__
    galois::opencl::CLContext * ctx = galois::opencl::getCLContext();
    cl_int err;
    dev_data= clCreateBuffer(ctx->get_default_device()->context(), CL_MEM_READ_WRITE, sizeof(Ty) , nullptr, &err);
    galois::opencl::CHECK_CL_ERROR(err, "Error allocating DGAccumulator!\n");
    Ty val = 0;
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &val, 0, NULL, NULL);
    galois::opencl::CHECK_CL_ERROR(err, "Error Writing DGAccumulator!\n");
#endif
  }
  DGAccumulator& operator+=(const Ty& rhs) {
    mdata += rhs;
    return *this;
  }
  /************************************************************
   *
   ************************************************************/

  void operator=(const Ty rhs) {
    mdata.reset();
    mdata += rhs;
#ifdef __GALOIS_HET_OPENCL__
    int err;
    galois::opencl::CLContext * ctx = galois::opencl::getCLContext();
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    Ty val = mdata.load();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &val, 0, NULL, NULL);
    galois::opencl::CHECK_CL_ERROR(err, "Error Writing DGAccumulator!\n");
#endif
  }

  /************************************************************
   *
   ************************************************************/

  void set(const Ty rhs) {
    mdata.reset();
    mdata += rhs;
#ifdef __GALOIS_HET_OPENCL__
    int err;
    galois::opencl::CLContext * ctx = galois::opencl::getCLContext();
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &mdata.load(), 0, NULL, NULL);
    galois::opencl::CHECK_CL_ERROR(err, "Error writing DGAccumulator!\n");
#endif
  }

  Ty read_local() {
    if (local_mdata == 0) local_mdata = mdata.reduce();
    return local_mdata;
  }

  Ty read() {
    return global_mdata;
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

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    switch (bare_mpi) {
      case noBareMPI:
#endif
        reduce_net();
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
        break;
      case nonBlockingBareMPI:
      case oneSidedBareMPI:
        reduce_mpi(local_mdata);
        break;
      default:
        GALOIS_DIE("Unsupported bare MPI");
    }
#endif

    reduceTimer.stop();

    return global_mdata;
  }


  /************************************************************
   *
   ************************************************************/

  /* Max reduction across DGAccumulators */
  Ty reduce_max() {
    if (local_mdata == 0) local_mdata = mdata.reduce();
    global_mdata = local_mdata;
#ifdef __GALOIS_HET_OPENCL__
    Ty tmp;
    galois::opencl::CLContext * ctx = galois::opencl::getCLContext();
    cl_int err = clEnqueueReadBuffer(ctx->get_default_device()->command_queue(), dev_data, CL_TRUE, 0, sizeof(Ty), &tmp, 0, NULL, NULL);
//    fprintf(stderr, "READ-DGA[%d, %d]\n", galois::runtime::NetworkInterface::ID, tmp);
    galois::opencl::CHECK_CL_ERROR(err, "Error reading DGAccumulator!\n");
    // TODO change to atomic max?
    galois::atomicAdd(mdata, tmp);
#endif
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
  Ty reduce_min() {
    if (local_mdata == 0) local_mdata = mdata.reduce();
    global_mdata = local_mdata;
#ifdef __GALOIS_HET_OPENCL__
    Ty tmp;
    galois::opencl::CLContext * ctx = galois::opencl::getCLContext();
    cl_int err = clEnqueueReadBuffer(ctx->get_default_device()->command_queue(), dev_data, CL_TRUE, 0, sizeof(Ty), &tmp, 0, NULL, NULL);
//    fprintf(stderr, "READ-DGA[%d, %d]\n", galois::runtime::NetworkInterface::ID, tmp);
    galois::opencl::CHECK_CL_ERROR(err, "Error reading DGAccumulator!\n");
    // TODO change to atomic min?
    galois::atomicAdd(mdata, tmp);
#endif
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

  /************************************************************
   *
   ************************************************************/

#ifdef __GALOIS_HET_OPENCL__
  const cl_mem &device_ptr(){
    reset();
    return dev_data;
  }
#endif
  /************************************************************
   *
   ************************************************************/
  Ty reset() {
    Ty retval = global_mdata;
    mdata.reset();
    local_mdata = global_mdata = 0;
#ifdef __GALOIS_HET_OPENCL__
    int err;
    Ty val = mdata.load();
    galois::opencl::CLContext * ctx = galois::opencl::getCLContext();
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &val, 0, NULL, NULL);
    galois::opencl::CHECK_CL_ERROR(err, "Error writing (reset) DGAccumulator!\n");
//    fprintf(stderr, "RESET-DGA[%d, %d]\n", galois::runtime::NetworkInterface::ID, val);
#endif
    return retval;
  }
};
}
#endif
