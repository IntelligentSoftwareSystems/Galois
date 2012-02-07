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
  unsigned getPackageForThread(int galois_thread_id);
  //find the maximum package number for all threads up to and including id
  unsigned getMaxPackageForThread(int galois_thread_id);

}
}

#endif //_HWTOPO_H
