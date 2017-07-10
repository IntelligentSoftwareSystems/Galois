/** Distributed Accumulator type -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"

#ifdef __GALOIS_HET_OPENCL__
#include "Galois/OpenCL/CL_Header.h"
#endif

namespace Galois {

template<typename Ty>
class DGAccumulator {
  Galois::Runtime::NetworkInterface& net = Galois::Runtime::getSystemNetworkInterface();

  Galois::GAccumulator<Ty> mdata;
  Ty local_mdata, global_mdata;
#ifdef __GALOIS_HET_OPENCL__
  cl_mem dev_data;
#endif

public:
  // Default constructor
  DGAccumulator() {
#ifdef __GALOIS_HET_OPENCL__
    Galois::OpenCL::CLContext * ctx = Galois::OpenCL::getCLContext();
    cl_int err;
    dev_data= clCreateBuffer(ctx->get_default_device()->context(), CL_MEM_READ_WRITE, sizeof(Ty) , nullptr, &err);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error allocating DGAccumulator!\n");
    Ty val = 0;
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &val, 0, NULL, NULL);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error Writing DGAccumulator!\n");
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
    Galois::OpenCL::CLContext * ctx = Galois::OpenCL::getCLContext();
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    Ty val = mdata.load();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &val, 0, NULL, NULL);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error Writing DGAccumulator!\n");
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
    Galois::OpenCL::CLContext * ctx = Galois::OpenCL::getCLContext();
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &mdata.load(), 0, NULL, NULL);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error writing DGAccumulator!\n");
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

  Ty reduce() {
    if (local_mdata == 0) local_mdata = mdata.reduce();
    global_mdata = local_mdata;
#ifdef __GALOIS_HET_OPENCL__
    Ty tmp;
    Galois::OpenCL::CLContext * ctx = Galois::OpenCL::getCLContext();
    cl_int err = clEnqueueReadBuffer(ctx->get_default_device()->command_queue(), dev_data, CL_TRUE, 0, sizeof(Ty), &tmp, 0, NULL, NULL);
//    fprintf(stderr, "READ-DGA[%d, %d]\n", Galois::Runtime::NetworkInterface::ID, tmp);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error reading DGAccumulator!\n");
    Galois::atomicAdd(mdata, tmp);
#endif
    for (unsigned h = 1; h < net.Num; ++h) {
      unsigned x = (net.ID + h) % net.Num;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, local_mdata);
      net.sendTagged(x, Galois::Runtime::evilPhase, b);
    }
    net.flush();

    unsigned num_Hosts_recvd = 1;
    while (num_Hosts_recvd < net.Num) {
      decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      } while (!p);
      Ty x_mdata;
      gDeserialize(p->second, x_mdata);
      global_mdata += x_mdata;
      ++num_Hosts_recvd;
    }
    ++Galois::Runtime::evilPhase;

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
    Galois::OpenCL::CLContext * ctx = Galois::OpenCL::getCLContext();
    cl_int err = clEnqueueReadBuffer(ctx->get_default_device()->command_queue(), dev_data, CL_TRUE, 0, sizeof(Ty), &tmp, 0, NULL, NULL);
//    fprintf(stderr, "READ-DGA[%d, %d]\n", Galois::Runtime::NetworkInterface::ID, tmp);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error reading DGAccumulator!\n");
    // TODO change to atomic max?
    Galois::atomicAdd(mdata, tmp);
#endif
    for (unsigned h = 1; h < net.Num; ++h) {
      unsigned x = (net.ID + h) % net.Num;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, local_mdata);
      net.sendTagged(x, Galois::Runtime::evilPhase, b);
    }
    net.flush();

    unsigned num_Hosts_recvd = 1;
    while (num_Hosts_recvd < net.Num) {
      decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      } while (!p);
      Ty x_mdata;
      gDeserialize(p->second, x_mdata);
      Galois::max(global_mdata, x_mdata);
      ++num_Hosts_recvd;
    }
    ++Galois::Runtime::evilPhase;

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
    Galois::OpenCL::CLContext * ctx = Galois::OpenCL::getCLContext();
    cl_int err = clEnqueueReadBuffer(ctx->get_default_device()->command_queue(), dev_data, CL_TRUE, 0, sizeof(Ty), &tmp, 0, NULL, NULL);
//    fprintf(stderr, "READ-DGA[%d, %d]\n", Galois::Runtime::NetworkInterface::ID, tmp);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error reading DGAccumulator!\n");
    // TODO change to atomic min?
    Galois::atomicAdd(mdata, tmp);
#endif
    for (unsigned h = 1; h < net.Num; ++h) {
      unsigned x = (net.ID + h) % net.Num;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, local_mdata);
      net.sendTagged(x, Galois::Runtime::evilPhase, b);
    }
    net.flush();

    unsigned num_Hosts_recvd = 1;
    while (num_Hosts_recvd < net.Num) {
      decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      } while (!p);
      Ty x_mdata;
      gDeserialize(p->second, x_mdata);
      Galois::min(global_mdata, x_mdata);
      ++num_Hosts_recvd;
    }
    ++Galois::Runtime::evilPhase;

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
    Galois::OpenCL::CLContext * ctx = Galois::OpenCL::getCLContext();
    cl_command_queue queue = ctx->get_default_device()->command_queue();
    err = clEnqueueWriteBuffer(queue, dev_data, CL_TRUE, 0, sizeof(Ty), &val, 0, NULL, NULL);
    Galois::OpenCL::CHECK_CL_ERROR(err, "Error writing (reset) DGAccumulator!\n");
//    fprintf(stderr, "RESET-DGA[%d, %d]\n", Galois::Runtime::NetworkInterface::ID, val);
#endif
    return retval;
  }
};
}
#endif
