/** Hardware topology and thread binding -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
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
 * @section Description
 *
 * Report HW topology and allow thread binding.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_SUBSTRATE_HWTOPO_H
#define GALOIS_SUBSTRATE_HWTOPO_H

#include <memory>

namespace Galois {
namespace Substrate {

class HWTopo {
public:

  struct threadInfo {
    unsigned tid;//tid == this thread
    unsigned packageLeader; // first thread id in tid's package
    unsigned package; // package of tid
    unsigned hwContext; // OS HW context numbering for bound thread
  };

  struct machineInfo {
    unsigned maxThreads;
    unsigned maxCores;
    unsigned maxPackages;
  };

  //! Get metadata for thread
  virtual const threadInfo& getThreadInfo(unsigned galois_thread_id) const = 0;
  //! Bind thread specified by id to the correct OS thread
  virtual bool bindThreadToProcessor(unsigned galois_thread_id) const = 0;
  //! get metadata for machine
  virtual const machineInfo& getMachineInfo() const = 0;

  //! Map thread to package
  //  virtual unsigned getPackageForThread(int galois_thread_id) const = 0;
  //! Find the maximum package number for all threads up to and including id
  //virtual unsigned getMaxPackageForThread(int galois_thread_id) const = 0;
  //! is this the first thread in a package
  //virtual bool isPackageLeader(int galois_thread_id) const = 0;

  //virtual unsigned getLeaderForThread(int galois_thread_id) const = 0;
  //virtual unsigned getLeaderForPackage(int galois_pkg_id) const = 0;

  unsigned getMaxPackageForThread(unsigned tid) const;

};

std::unique_ptr<HWTopo> getHWTopo();

// extern __thread unsigned PACKAGE_ID;

// static inline unsigned fillPackageID(int galois_thread_id) {
//   unsigned x = getPackageForThread(galois_thread_id);
//   bool y = isPackageLeader(galois_thread_id);
//   x = (x << 2) | ((y ? 1 : 0) << 1) | 1;
//   PACKAGE_ID = x;
//   return x;
// }

// //! Optimized when galois_thread_id corresponds to the executing thread
// static inline unsigned getPackageForSelf(int galois_thread_id) {
//   unsigned x = PACKAGE_ID;
//   if (x & 1)
//     return x >> 2;
//   x = fillPackageID(galois_thread_id);
//   return x >> 2;
// }

// //! Optimized when galois_thread_id corresponds to the executing thread
// static inline bool isPackageLeaderForSelf(int galois_thread_id) {
//   unsigned x = PACKAGE_ID;
//   if (x & 1)
//     return (x >> 1) & 1;
//   x = fillPackageID(galois_thread_id);
//   return (x >> 1) & 1;
// }

} // end namespace Substrate
} // end namespace Galois

#endif //_HWTOPO_H
