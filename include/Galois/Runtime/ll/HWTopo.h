/** Hardware topology and thread binding -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights
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

#ifndef _HWTOPO_H
#define _HWTOPO_H

namespace GaloisRuntime {
namespace LL {

//Bind thread specified by id to the correct OS thread
bool bindThreadToProcessor(int galois_thread_id);
//Get number of threads supported
unsigned getMaxThreads();
//get number of cores supported
unsigned getMaxCores();
//get number of packages supported
unsigned getMaxPackages();
//map thread to package
unsigned getPackageForThreadInternal(int galois_thread_id);
//find the maximum package number for all threads up to and including id
unsigned getMaxPackageForThread(int galois_thread_id);
//is This the first thread in a package
bool isLeaderForPackageInternal(int galois_thread_id);

extern __thread unsigned PACKAGE_ID;

static inline unsigned fillPackageID(int galois_thread_id) {
  unsigned x = getPackageForThreadInternal(galois_thread_id);
  bool y = isLeaderForPackageInternal(galois_thread_id);
  x = (x << 2) | ((y ? 1 : 0) << 1) | 1;
  PACKAGE_ID = x;
  return x;
}

static inline unsigned getPackageForThread(int galois_thread_id) {
  unsigned x = PACKAGE_ID;
  if (x & 1)
    return x >> 2;
  x = fillPackageID(galois_thread_id);
  return x >> 2;
}

static inline bool isLeaderForPackage(int galois_thread_id) {
  unsigned x = PACKAGE_ID;
  if (x & 1)
    return (x >> 1) & 1;
  x = fillPackageID(galois_thread_id);
  return (x >> 1) & 1;
}

}
}

#endif //_HWTOPO_H
