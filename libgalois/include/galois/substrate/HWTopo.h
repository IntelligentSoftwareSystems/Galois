/** Hardware topology and thread binding -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * @section Description
 *
 * Report HW topology and allow thread binding.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_SUBSTRATE_HWTOPO_H
#define GALOIS_SUBSTRATE_HWTOPO_H

#include <vector>

namespace galois {
namespace substrate {

struct threadTopoInfo {
  unsigned tid; // this thread (galois id)
  unsigned socketLeader; //first thread id in tid's package
  unsigned socket; // socket (L3 normally) of thread
  unsigned numaNode; // memory bank.  may be different than socket.
  unsigned cumulativeMaxSocket; // max package id seen from [0, tid]
  unsigned osContext; // OS ID to use for thread binding
  unsigned osNumaNode; // OS ID for numa node
};

struct machineTopoInfo {
  unsigned maxThreads;
  unsigned maxCores;
  unsigned maxPackages;
  unsigned maxNumaNodes;
};

//parse machine topology
std::pair<machineTopoInfo,std::vector<threadTopoInfo>> getHWTopo();
//bind a thread to a hwContext (returned by getHWTopo)
bool bindThreadSelf(unsigned osContext);


} // end namespace substrate
} // end namespace galois

#endif //_HWTOPO_H
