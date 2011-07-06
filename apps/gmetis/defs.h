/** GMetis -*- C++ -*-
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
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
