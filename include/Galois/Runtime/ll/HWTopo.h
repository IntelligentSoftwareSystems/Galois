/** Hardware topology and thread binding -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights
 * reserved.  UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE,
 * NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY WARRANTY
 * THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF
 * TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO
 * THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect,
 * direct or consequential damages or loss of profits, interruption of
 * business, or related expenses which may arise from use of Software
 * or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of
 * data of any kind.  
 *
 * @section Description
 *
 * Report HW topology and allow thread binding.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_LL_HWTOPO_H
#define GALOIS_RUNTIME_LL_HWTOPO_H

namespace GaloisRuntime {
namespace LL {

//! Bind thread specified by id to the correct OS thread
bool bindThreadToProcessor(int galois_thread_id);
//! Get number of threads supported
unsigned getMaxThreads();
//! Get number of cores supported
unsigned getMaxCores();
//! Get number of packages supported
unsigned getMaxPackages();
//! Map thread to package
unsigned getPackageForThread(int galois_thread_id);
//! Find the maximum package number for all threads up to and including id
unsigned getMaxPackageForThread(int galois_thread_id);
//! is this the first thread in a package
bool isPackageLeader(int galois_thread_id);

unsigned getLeaderForThread(int galois_thread_id);
unsigned getLeaderForPackage(int galois_pkg_id);

extern __thread unsigned PACKAGE_ID;

static inline unsigned fillPackageID(int galois_thread_id) {
  unsigned x = getPackageForThread(galois_thread_id);
  bool y = isPackageLeader(galois_thread_id);
  x = (x << 2) | ((y ? 1 : 0) << 1) | 1;
  PACKAGE_ID = x;
  return x;
}

//! Optimized when galois_thread_id corresponds to the executing thread
static inline unsigned getPackageForSelf(int galois_thread_id) {
  unsigned x = PACKAGE_ID;
  if (x & 1)
    return x >> 2;
  x = fillPackageID(galois_thread_id);
  return x >> 2;
}

//! Optimized when galois_thread_id corresponds to the executing thread
static inline bool isPackageLeaderForSelf(int galois_thread_id) {
  unsigned x = PACKAGE_ID;
  if (x & 1)
    return (x >> 1) & 1;
  x = fillPackageID(galois_thread_id);
  return (x >> 1) & 1;
}

}
}

#endif //_HWTOPO_H
