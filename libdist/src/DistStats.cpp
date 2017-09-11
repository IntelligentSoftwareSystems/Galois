/** DistStatManager Implementation -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author M. Amber Hassaan<ahassaan@ices.utexas.edu>
 */

#include "Galois/DistStats.h"

using namespace Galois::Runtime;

static void recvAtHost_0_int(uint32_t hostID, const Galois::gstl::String loopname, const Galois::gstl::String category, const Galois::gstl::Vector<int64_t> threadVals) {
  getSM()->addDistStat_int(hostID, loopname, category, threadVals);
}

static void recvAtHost_0_fp(uint32_t hostID, const Galois::gstl::String loopname, const Galois::gstl::String category, const Galois::gstl::Vector<double> threadVals) {
  getSM()->addDistStat_fp(hostID, loopname, category, threadVals);
}

void DistStatManager::combineAtHost_0(void) {
  Galois::Runtime::getHostBarrier().wait();

  if (getHostID() != 0) {
    for (auto i = Base::intBegin(), end_i = Base::intEnd(); i != end_i; ++i) {
      Str ln;
      Str cat;
      Galois::gstl::Vector<int64_t> threadVals;

      Base::readStat(i, ln, cat, threadVals);

      getSystemNetworkInterface().sendSimple(0, recvAtHost_0_int, ln, cat, threadVals());
    }

    for (auto i = Base::fpBegin(), end_i = Base::fpEnd(); i != end_i; ++i) {
      Str ln;
      Str cat;
      Galois::gstl::Vector<double> threadVals;

      Base::getStat(i, ln, cat, threadVals);

      getSystemNetworkInterface().sendSimple(0, recvAtHost_0_fp, ln, cat, threadVals());
    }
  }

  Galois::Runtime::getHostBarrier().wait();
}


