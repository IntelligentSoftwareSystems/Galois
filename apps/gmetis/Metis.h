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
 * @author Nikunj Yadav <nikunj@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef METIS_H_
#define METIS_H_


struct partInfo {
  unsigned partNum;
  unsigned partMask;
  unsigned partWeight;
  unsigned totalWeight;
  unsigned partSize; 
  int neededParts; 
  GGraph* graph;
  int tryPart;
  
  //  partInfo(GGraph* g, unsigned tw, unsigned mw = 0)
  // :partNum(0), partMask(1), partWeight(mw), totalWeight(tw), graph(g) {}
  partInfo(GGraph* g,unsigned size, unsigned tw, int tryN, int neededParts=1, unsigned mw = 0)
    :partNum(0), partMask(1), partWeight(mw), totalWeight(tw), partSize(size), neededParts(neededParts), graph(g), tryPart(tryN) { 
  } 

  partInfo() { } 

  partInfo split() {
    partInfo np(*this);
    np.partNum = partNum | partMask;
    partMask <<= 1;
    np.partMask = partMask;
    np.neededParts = neededParts/2;
    neededParts -= np.neededParts;
    np.partWeight = 0;
    np.tryPart = tryPart;
    return np;
  }
};

std::ostream& operator<<(std::ostream& os, const partInfo& p);

struct parallelBisect { 
  //bisect_policy bisect; 
  partInfo** parts; 
  parallelBisect(partInfo** parts);  
  void operator()(partInfo &item, Galois::UserContext<partInfo> &workList);
}; 


//Coarsening
MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo);

//Partitioning
partInfo bisect_GGP(partInfo& oldPart);
partInfo bisect_GGGP(partInfo& oldPart);


//Refinement
void refine_FL_pair(partInfo& p1, partInfo& p2);
void refine_BKL(GGraph& g, std::vector<partInfo>& parts);

#endif
