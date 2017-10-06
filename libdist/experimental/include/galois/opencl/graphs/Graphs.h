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
}
;
#endif
#include "galois/opencl/graphs/GraphUtils.h"
#include "galois/opencl/graphs/CL_LC_Graph.h"
#include "galois/opencl/graphs/CL_LC_VoidGraph.h"
//#include "LC_GraphVoid_2.h"
//#include "LC_LinearArray_Graph.h"
//#include "LC_LinearArray_VoidGraph.h"

#endif /* GDIST_CL_GRAPHS_H_ */
