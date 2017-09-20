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
#include <OpenCL/opencl.h>
#else
extern "C" {
#include "CL/cl.h"
}
;
#endif
#include "galois/OpenCL/graphs/GraphUtils.h"
#include "galois/OpenCL/graphs/CL_LC_Graph.h"
#include "galois/OpenCL/graphs/CL_LC_VoidGraph.h"
//#include "LC_GraphVoid_2.h"
//#include "LC_LinearArray_Graph.h"
//#include "LC_LinearArray_VoidGraph.h"

#endif /* GDIST_CL_GRAPHS_H_ */
