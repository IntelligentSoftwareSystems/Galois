#ifndef _TET_LINEAR_COORD_CONN_H_
#define _TET_LINEAR_COORD_CONN_H_

#include "AuxDefs.h"

/**
 * important constants for linear tetrahedron
 */
struct TetLinearTraits {
  enum {
    SPD = 3,
    NODES_PER_ELEM = 4,
    TOPO = 6,
    NUM_EDGES = 6,
    NFIELDS = SPD,
  };
};

class TetLinearCoordConn
: public AbstractCoordConn <TetLinearTraits::SPD, TetLinearTraits::NODES_PER_ELEM, TetLinearTraits::TOPO> {
  public:
    static const size_t NUM_EDGES = TetLinearTraits::NUM_EDGES;

  protected:
    /**
     * Return an instance of 3D with linear shape functions
     * and linear tetrahedron as geometry
     */
    virtual Element* makeElem (const size_t elemIndex) const {
      VecSize_t conn;

      genElemConnectivity (elemIndex, conn);

      Tetrahedron* tetGeom = new Tetrahedron (coordinates, conn);
      return new P13D<TetLinearTraits::NFIELDS>::Bulk (*tetGeom);
    }

  private:
    /**
     *
     * @param neighbors: the output vector
     * for each element p, populate an indexed list L of pairs (q,j), where i is index of each pair,
     * such that
     * p shares it's i with q's edge j.
     * There should be a corresponding entry (p,i) in q's list at index j.
     *
     */

    void getEdgeNeighborList (std::vector<std::vector<std::vector<size_t> > > &neighbors) const {

      size_t iElements = getNumElements ();

      neighbors.clear();
      neighbors.resize(iElements);

      std::vector<edgestruct> edges;

      size_t V1[] = { 0, 1, 0, 2, 0, 1 };
      size_t V2[] = { 1, 2, 2, 3, 3, 3 };

      // the 4 nodes of a tet are numbered 0..3
      // edges are 0-1, 0-2, 0-3, 1-2, 1-3, 2-3
      // the nodes corresponding to edges are picked up in an involved manner using two arrays V1 and V2
      // the edges must be sorted in a consistent manner, due to which the nodes in an edge must be
      // sorted.

      // Creating a list of all possible edges.
      for (size_t e = 0; e < iElements; e++) {
        neighbors[e].resize(NUM_EDGES);

        const size_t econn[] = { connectivity[e * 4 + 0], connectivity[e * 4 + 1], connectivity[e * 4 + 2], connectivity[e * 4 + 3] };

        for (size_t edgenum = 0; edgenum < NUM_EDGES; edgenum++) {
          GlobalNodalIndex node0;
          GlobalNodalIndex node1;

          node0 = econn[V1[edgenum]];
          node1 = econn[V2[edgenum]];
          edgestruct myedge(e, edgenum, node0, node1);
          edges.push_back(myedge);
        }
      }

      std::sort(edges.begin(), edges.end());

      // Edges that have exactly the same connectivity should appear consecutively.
      // If there is no repetition, the edgeId is free.
      std::vector<edgestruct>::iterator it1 = edges.begin();
      while (it1 != edges.end()) {
        std::vector<edgestruct> repedges;
        repedges.clear();
        repedges.push_back(*it1);
        std::vector<edgestruct>::iterator it2 = it1 + 1;

        while (true && it2 != edges.end()) {
          // check for same connectivity
          if ((it2->node0 == it1->node0) && (it2->node1 == it1->node1)) {
            repedges.push_back(*it2);
            it2++;

          } else {
            break;
          }
        }

        if (repedges.size() > 1) { // Shared edgeId.
          for (size_t p = 0; p < repedges.size(); p++) {
            for (size_t q = 0; q < repedges.size(); q++) {
              if (p != q) {
                neighbors[repedges[p].elemId][repedges[p].edgeId]. push_back(repedges[q].elemId);

                neighbors[repedges[p].elemId][repedges[p].edgeId]. push_back(repedges[q].edgeId);
              }
            }
          }
        }

        it1 = it2;
      }
      // done.
    }

  public:

    /**
     * Purpose : Subdivide a tetrahedron in 8 smaller ones.
     * Algorithm to subdivide a test:
     * Parent tet: ABCD.
     * Since a consistent numbering of edges is crucial, the following convention
     * is adopted : 1 - AB, 2-BC, 3-CA, 4-CD, 5-AD, 6-BD.
     * Midpoints of edges AB,BC,CA,CD,AD,BD are M1, M2, M3, M4, M5, M6 resply.

     * Tet1: A-M1-M3-M5,
     * Tet2: M1-B-M2-M6,
     * Tet3: M3-M2-C-M4,
     * Tet4: M5-M6-M4-D,

     * Tet5: M1-M4-M5-M6,
     * Tet6: M1-M4-M6-M2,
     * Tet7: M1-M4-M2-M3,
     * Tet8: M1-M4-M3-M5.
     */
    virtual void subdivide () {

      size_t sd = getSpatialDim();
      size_t eNodes = getNodesPerElem(); // Number of nodes per element.

      size_t iElements = getNumElements(); // Number of elements.
      size_t iNodes = getNumNodes();

      std::vector<std::vector<std::vector<size_t> > > neighbors;
      getEdgeNeighborList(neighbors);

      // Connectivity for mid-points of each edgeId for each element.
      // size_t midconn[iElements][NUM_EDGES];

      std::vector<std::vector<size_t> > midconn (iElements);
      for (size_t i = 0; i < midconn.size (); ++i) {
        midconn[i].resize (NUM_EDGES);
      }

      size_t count = iNodes;

      for (size_t e = 0; e < iElements; e++) {
        for (size_t f = 0; f < NUM_EDGES; f++) {

          // Number of elements sharing edgeId 'f' of element 'e'.
          size_t nNeighbors = neighbors[e][f].size() / 2;

          if (nNeighbors == 0) { // Free edgeId.

            // for 0-based node numbering we increment 'count' afterwards
            // count++;
            midconn[e][f] = count;
            ++count;

          } else { // Shared edgeId
            // Find the least element neighbor number.
            size_t minElem = e;
            for (size_t p = 0; p < nNeighbors; p++) {
              if (minElem > neighbors[e][f][2 * p]) {
                minElem = neighbors[e][f][2 * p];
              }
            }

            if (e == minElem) { // Allot only once for a shared edgeId.
              // for 0-based node numbering we increment 'count' afterwards
              // count++;
              midconn[e][f] = count;

              for (size_t p = 0; p < nNeighbors; p++) {
                size_t nelem = neighbors[e][f][2 * p];
                size_t nedge = neighbors[e][f][2 * p + 1];
                midconn[nelem][nedge] = count;
              }
              // increment 'count' now
              ++count;
            }
          }
        }
      }

      // Creating new coordinates and connectivity arrays:
      // Each tet is subdivided into 8.
      VecDouble newCoord(count * sd);
      std::vector<size_t> newConn;

      for (size_t i = 0; i < coordinates.size(); i++) {
        newCoord[i] = coordinates[i];
      }

      // Coordinates for midside nodes:
      size_t V1[] = { 0, 1, 0, 2, 0, 1 };
      size_t V2[] = { 1, 2, 2, 3, 3, 3 };
      for (size_t e = 0; e < iElements; e++)
        for (size_t f = 0; f < NUM_EDGES; f++) {
          // for 0-based node numbering, we don't need to subtract 1 from node ids in connectivity
          // size_t v1 = connectivity[e * eNodes + V1[f]] - 1;
          // size_t v2 = connectivity[e * eNodes + V2[f]] - 1;
          // for (size_t k = 0; k < sd; k++)
          // newCoord[(midconn[e][f] - 1) * sd + k] = 0.5 * (coordinates[v1 * sd + k] + coordinates[v2 * sd + k]);
          size_t v1 = connectivity[e * eNodes + V1[f]];
          size_t v2 = connectivity[e * eNodes + V2[f]];
          for (size_t k = 0; k < sd; k++) {
            newCoord[midconn[e][f] * sd + k] = 0.5 * (coordinates[v1 * sd + k] + coordinates[v2 * sd + k]);
          }
        }

      for (size_t e = 0; e < iElements; e++) {
        // tet 1-8
        // four at conrners
        size_t t1conn[] = { connectivity[e * eNodes + 0], midconn[e][0], midconn[e][2], midconn[e][4] };
        size_t t2conn[] = {  midconn[e][0], connectivity[e * eNodes + 1], midconn[e][1], midconn[e][5] };
        size_t t3conn[] = { midconn[e][2], midconn[e][1], connectivity[e * eNodes + 2], midconn[e][3] };
        size_t t4conn[] = { midconn[e][4], midconn[e][5], midconn[e][3], connectivity[e * eNodes + 3] };

        // four in the middle
        size_t t5conn[] = { midconn[e][0], midconn[e][3], midconn[e][4], midconn[e][5] };
        size_t t6conn[] = { midconn[e][0], midconn[e][3], midconn[e][5], midconn[e][1] };
        size_t t7conn[] = { midconn[e][0], midconn[e][3], midconn[e][1], midconn[e][2] };
        size_t t8conn[] = { midconn[e][0], midconn[e][3], midconn[e][2], midconn[e][4] };

        newConn.insert (newConn.end (), &t1conn[0], &t1conn[eNodes]);
        newConn.insert (newConn.end (), &t2conn[0], &t2conn[eNodes]);
        newConn.insert (newConn.end (), &t3conn[0], &t3conn[eNodes]);
        newConn.insert (newConn.end (), &t4conn[0], &t4conn[eNodes]);
        newConn.insert (newConn.end (), &t5conn[0], &t5conn[eNodes]);
        newConn.insert (newConn.end (), &t6conn[0], &t6conn[eNodes]);
        newConn.insert (newConn.end (), &t7conn[0], &t7conn[eNodes]);
        newConn.insert (newConn.end (), &t8conn[0], &t8conn[eNodes]);

      }
      coordinates.clear();
      connectivity.clear();
      coordinates.assign(newCoord.begin(), newCoord.end());
      connectivity.assign(newConn.begin(), newConn.end());

      // nodes = size_t(coordinates.size() / 3);
      // elements = size_t(connectivity.size() / 4);
    }
};

#endif // _TET_LINEAR_COORD_CONN_H_
