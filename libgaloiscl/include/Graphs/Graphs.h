/*
 * Graphs.h
 *
 *  Created on: Feb 18, 2016
 *      Author: rashid
 */

#ifndef GDIST_LIBGALOISCL_INCLUDE_GRAPHS_GRAPHS_H_
#define GDIST_LIBGALOISCL_INCLUDE_GRAPHS_GRAPHS_H_

//#include "CL_LC_Graph.h.bak"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
extern "C" {
#include "CL/cl.h"
}
;
#endif
#include "GraphUtils.h"
#include "CL_LC_Graph.h"
//#include "LC_GraphVoid_2.h"
//#include "LC_LinearArray_Graph.h"
//#include "LC_LinearArray_VoidGraph.h"


#endif /* GDIST_LIBGALOISCL_INCLUDE_GRAPHS_GRAPHS_H_ */
