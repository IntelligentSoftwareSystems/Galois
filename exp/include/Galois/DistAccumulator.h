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

namespace Galois {

  template<typename Ty>
  class DGAccumulator {
    Galois::Runtime::NetworkInterface& net = Galois::Runtime::getSystemNetworkInterface();

    std::atomic<Ty> mdata;
    static Ty others_mdata;
    static unsigned num_Hosts_recvd;

    public:
      DGAccumulator& operator+=(const Ty& rhs){
        Galois::atomicAdd(mdata, rhs);
        return *this;
      }

      void operator=(const Ty rhs){
        mdata.store(rhs);
      }

      void set(const Ty rhs){
        mdata.store(rhs);
      }
      static void reduce_landingPad(Galois::Runtime::RecvBuffer& buf){
        uint32_t x_id;
        Ty x_mdata;
        gDeserialize(buf, x_id, x_mdata);
        others_mdata += x_mdata;
        ++num_Hosts_recvd;
      }

      Ty reduce(){
        for(auto x = 0; x < net.Num; ++x){
          if(x == net.ID)
            continue;
          Galois::Runtime::SendBuffer b;
          gSerialize(b, net.ID, mdata);
          net.send(x, reduce_landingPad, b);
        }

        net.flush();
        while(num_Hosts_recvd < (net.Num - 1)){
          net.handleReceives();
        }
        Galois::Runtime::getHostBarrier().wait();

        Galois::atomicAdd(mdata, others_mdata);
        others_mdata = 0;
        num_Hosts_recvd = 0;
        return mdata;
      }

      Ty reset(){
        return mdata.exchange(0);
      }

  };

  template<typename Ty>
  Ty DGAccumulator<Ty>::others_mdata;

  template<typename Ty>
  unsigned DGAccumulator<Ty>::num_Hosts_recvd = 0;
}
#endif
