/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

/*
 * Node.cpp
 *
 *  Created on: Aug 30, 2013
 *      Author: kjopek
 */

#include "Node.h"
#include "Production.h"
#include "EProduction.hxx"

#include <galois/Galois.h>

#include <sys/time.h>
#include <sched.h>

void Node::execute() {

  productions->Execute(productionToExecute, v, input);
  // struct timeval t1, t2;

  // gettimeofday(&t1, NULL);

  // int tid = galois::runtime::LL::getTID();

  // gettimeofday(&t2, NULL);

  // printf("Production: %d executed on [%d / %d] in: %f [s]\n",
  //		productionToExecute, tid,
  //galois::runtime::LL::getSocketForSelf(tid),
  //		((t2.tv_sec-t1.tv_sec) * 1e6 + (t2.tv_usec-t1.tv_usec))/1e6
  //		);
}
