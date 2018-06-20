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
 * Graphs.h
 *
 *  Created on: Feb 18, 2016
 *      Author: rashid
 */

#ifndef GDIST_CL_GRAPHS_H_
#define GDIST_CL_GRAPHS_H_

//#include "CL_LC_Graph.h.bak"
#ifdef __APPLE__
#include <opencl/opencl.h>
#else
extern "C" {
#include "CL/cl.h"
};
#endif
#include "galois/opencl/graphs/GraphUtils.h"
#include "galois/opencl/graphs/CL_LC_Graph.h"
#include "galois/opencl/graphs/CL_LC_VoidGraph.h"
//#include "LC_GraphVoid_2.h"
//#include "LC_LinearArray_Graph.h"
//#include "LC_LinearArray_VoidGraph.h"

#endif /* GDIST_CL_GRAPHS_H_ */
