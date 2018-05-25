/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef _TRI_LINEAR_COORD_CONN_H_
#define _TRI_LINEAR_COORD_CONN_H_

#include "CoordConn.h"

/**
 * important constants for linear triangle
 */
struct TriLinearTraits {
  enum {
    SPD = 2,
    NODES_PER_ELEM = 3,
    TOPO = 2,
    NUM_EDGES = 3,
    NFIELDS = SPD,
  };
};

class TriLinearCoordConn
: public AbstractCoordConn <TriLinearTraits::SPD, TriLinearTraits::NODES_PER_ELEM, TriLinearTraits::TOPO> {

  protected:
    /**
     *
     * Return a 2d element with linear shape functions and linear triangle as the geometry
     *
     * @param elemIndex
     */

    virtual Element* makeElem (const size_t elemIndex) const {
      VecSize_t conn;

      genElemConnectivity (elemIndex, conn);

      Triangle<TriLinearTraits::SPD>* triGeom = new Triangle<TriLinearTraits::SPD> (coordinates, conn);
      return new P12D<TriLinearTraits::NFIELDS>::Bulk (*triGeom);
    }

  public:

    /**
     * divides each triangle in the mesh in to 4 triangles
     * The main idea is to join the mid points of the three segments (called edges here)
     */
    virtual void subdivide () {
      // Check for consistency of connectivity and coordinates arrays:
      std::vector<edgestruct> faces;

      size_t iElements = getNumElements(); // 3 nodes per element.
      size_t iNodes = getNumNodes(); // Assume 2D triangles.

      for (size_t e = 0; e < iElements; e++) {
        // VecSize_t conn;
        GlobalNodalIndex node0;
        GlobalNodalIndex node1;


        node0 = connectivity[e * 3 + 0];
        node1 = connectivity[e * 3 + 1];
        faces.push_back (edgestruct (e, 0, node0, node1));

        node0 = connectivity[e * 3 + 1];
        node1 = connectivity[e * 3 + 2];
        faces.push_back (edgestruct (e, 1, node0, node1));

        node0 = connectivity[e * 3 + 2];
        node1 = connectivity[e * 3 + 0];
        faces.push_back (edgestruct (e, 2, node0, node1));
      }

      std::sort (faces.begin (), faces.end ());

      VecSize_t NodeInfo (faces.size () * 2, 0);
      size_t middlenodenum = iNodes;

      // Create middle nodes
      for (std::vector<edgestruct>::iterator it = faces.begin (); it != faces.end (); it++) {
        // for 1-based node numbering
        // double xm = (coordinates[2 * (it->conn[0] - 1)] + coordinates[2 * (it->conn[1] - 1)]) / 2.;
        // double ym = (coordinates[2 * (it->conn[0] - 1) + 1] + coordinates[2 * (it->conn[1] - 1) + 1]) / 2.;
        // 
        // for 0-based node numbering, we don't need to subtract 1 from conn field of facestruct
        double xm = (coordinates[2 * it->node0] + coordinates[2 * it->node1]) / 2.;
        double ym = (coordinates[2 * it->node0 + 1] + coordinates[2 * it->node1 + 1]) / 2.;
        coordinates.push_back (xm);
        coordinates.push_back (ym);

        NodeInfo[it->elemId * 6 + it->edgeId * 2 + 0] = it->edgeId;
        // for 0-based node numbering, don't add 1
        // NodeInfo[it->elemId * 6 + it->edgeId * 2 + 1] = middlenodenum + 1;
        NodeInfo[it->elemId * 6 + it->edgeId * 2 + 1] = middlenodenum;

        if (it + 1 != faces.end ()) {
          if (it->node0 == (it + 1)->node0 && it->node1 == (it + 1)->node1) {
            it++;
            NodeInfo[it->elemId * 6 + it->edgeId * 2 + 0] = it->edgeId;
            // for 0-based node numbering, don't add 1
            // NodeInfo[it->elemId * 6 + it->edgeId * 2 + 1] = middlenodenum + 1;
            NodeInfo[it->elemId * 6 + it->edgeId * 2 + 1] = middlenodenum;
          }
        }

        ++middlenodenum;
      }

      // Create connectivity
      VecSize_t copyconn (connectivity);
      connectivity.resize (iElements * 3 * 4);

      for (size_t e = 0; e < iElements; e++) {
        // triangle 1
        connectivity[e * 4 * 3 + 0 * 3 + 0] = copyconn[e * 3];
        connectivity[e * 4 * 3 + 0 * 3 + 1] = NodeInfo[e * 6 + 0 * 2 + 1];
        connectivity[e * 4 * 3 + 0 * 3 + 2] = NodeInfo[e * 6 + 2 * 2 + 1];

        // triangle 2
        connectivity[e * 4 * 3 + 1 * 3 + 0] = copyconn[e * 3 + 1];
        connectivity[e * 4 * 3 + 1 * 3 + 1] = NodeInfo[e * 6 + 1 * 2 + 1];
        connectivity[e * 4 * 3 + 1 * 3 + 2] = NodeInfo[e * 6 + 0 * 2 + 1];

        // triangle 3
        connectivity[e * 4 * 3 + 2 * 3 + 0] = copyconn[e * 3 + 2];
        connectivity[e * 4 * 3 + 2 * 3 + 1] = NodeInfo[e * 6 + 2 * 2 + 1];
        connectivity[e * 4 * 3 + 2 * 3 + 2] = NodeInfo[e * 6 + 1 * 2 + 1];

        // triangle 4
        connectivity[e * 4 * 3 + 3 * 3 + 0] = NodeInfo[e * 6 + 0 * 2 + 1];
        connectivity[e * 4 * 3 + 3 * 3 + 1] = NodeInfo[e * 6 + 1 * 2 + 1];
        connectivity[e * 4 * 3 + 3 * 3 + 2] = NodeInfo[e * 6 + 2 * 2 + 1];
      }

      // nodes = int(coordinates.size() / 2);
      // elements = int(connectivity.size() / 3);
    }

};

#endif // _TRI_LINEAR_COORD_CONN_H_
