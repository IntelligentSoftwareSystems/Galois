#ifndef _KRUSKAL_SERIAL_H_
#define _KRUSKAL_SERIAL_H_

#include "kruskalData.h"
#include "kruskalFunc.h"
#include "kruskal.h"

class KruskalSerial: public Kruskal<KNode> {
protected:

  virtual const std::string getVersion () const { return "Serial Ordered Kruskal"; }

  virtual void runMST (VecKNode_ty& nodes, VecKEdge_ty& edges,
      size_t& mstWeight, size_t& totalIter) {


    std::sort (edges.begin (), edges.end (), KEdge<KNode>::PtrComparator ());

    size_t mstSum = 0;
    size_t iter = 0;
    size_t numUnions = 0;

    for (VecKEdge_ty::const_iterator i = edges.begin (), ei = edges.end ();
        i != ei; ++i) {

      ++iter;

      if (kruskal::contract (**i)) {
        ++numUnions;
        mstSum += (*i)->weight;
      }
      

      if (numUnions == (nodes.size () - 1)) {
        break;
      }

    }

    mstWeight = mstSum;
    totalIter = iter;

  }

};

#endif // _KRUSKAL_SERIAL_H_
