/*
 * defs.h
 *
 *  Created on: Jun 21, 2011
 *      Author: xinsui
 */

#ifndef DEFS_H_
#define DEFS_H_
#include "MetisGraph.h"
void fmTwoWayEdgeRefine(MetisGraph* metisGraph, int* tpwgts, int npasses);
void balanceTwoWay(MetisGraph* metisGraph, int* tpwgts) ;
void refineKWay(MetisGraph* metisGraph, MetisGraph* orgGraph, float* tpwgts, float ubfactor, int nparts);
void greedyKWayEdgeBalance(MetisGraph* metisGraph, int nparts, float* tpwgts, float ubfactor,
		int npasses);
void bisection(MetisGraph* metisGraph, int* tpwgts, int coarsenTo);
void refineTwoWay(MetisGraph* metisGraph, MetisGraph* orgGraph, int* tpwgts);

#endif /* DEFS_H_ */
