/** Enviroment Checking Code -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_LL_ENVCHECK_H
#define GALOIS_RUNTIME_LL_ENVCHECK_H

namespace GaloisRuntime {
namespace LL {

//PLEASE document all enviroment variables here;
//ThreadPool_pthread.cpp: "GALOIS_DO_NOT_BIND_MAIN_THREAD"
//HWTopoLinux.cpp: "GALOIS_DEBUG_TOPO"
//Sampling.cpp: "GALOIS_EXIT_BEFORE_SAMPLING"
//Sampling.cpp: "GALOIS_EXIT_AFTER_SAMPLING"

//! Return true if the Enviroment variable is set
bool EnvCheck(const char* parm);
bool EnvCheck(const char* parm, int& val);

}
}

#endif
