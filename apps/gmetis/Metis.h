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
  unsigned partSize; 
  
  partInfo(unsigned mw, unsigned ms)
    :partNum(0), partMask(1), partWeight(mw), partSize(ms) {}

  partInfo() :partNum(~0), partMask(~0), partWeight(~0), partSize(~0) {}

  partInfo(unsigned pn, unsigned pm, unsigned pw, unsigned ps) :partNum(pn), partMask(pm), partWeight(pw), partSize(ps) {}

  unsigned splitID() const {
    return partNum | partMask;
  }

  std::pair<unsigned, unsigned> splitRatio(unsigned numParts) {
    unsigned L = 0, R = 0;
    unsigned LM = partMask - 1; // 00100 -> 00011
    for (unsigned x = 0; x < numParts; ++x)
      if ((x & LM) == partNum) {
        if (x & partMask)
          ++R;
        else
          ++L;
      }
    return std::make_pair(L, R);
  }

  partInfo split() {
    partInfo np(splitID(), partMask << 1, 0, 0);
    partMask <<= 1;
    return np;
  }
};

std::ostream& operator<<(std::ostream& os, const partInfo& p);

//Coarsening
MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo);

//Partitioning
std::vector<partInfo> partition(MetisGraph* coarseMetisGraph, unsigned numPartitions);

//Refinement
void refine_FL_pair(partInfo& p1, partInfo& p2);
void refine_BKL(GGraph& g, std::vector<partInfo>& parts);

#endif
