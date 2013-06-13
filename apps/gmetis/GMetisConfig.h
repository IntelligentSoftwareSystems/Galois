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

#ifndef GMETISCONFIG_H_
#define GMETISCONFIG_H_

#include "MetisNode.h"

#include "Galois/Graph/Graph.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Accumulator.h"
#include <stdlib.h>
//#include "Galois/Graph/LCGraph.h"
#include "Galois/Graph/LC_Morph_Graph.h"


/*
 *I imagine this might become a graph where instead of int being the
 *edge we might be storing a struct of int and pair<int,int> The first
 *int will be the normal weight, second would be a pair specifying
 *partition number and weight related to that partition.
 */
typedef Galois::Graph::LC_Morph_Graph<MetisNode,int> GGraph;
typedef Galois::Graph::LC_Morph_Graph<MetisNode,int>::GraphNode GNode;

int getRandom(int num);
int intlog2(int a);
#endif /* GMETISCONFIG_H_ */
