#ifndef _TET_LINEAR_COORD_CONN_H_
#define _TET_LINEAR_COORD_CONN_H_

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
    static const int NUM_EDGES = TetLinearTraits::NUM_EDGES;

  protected:
    virtual Element* makeElem (const size_t elemIndex) const {
      std::vector<GlobalNodalIndex> conn;

      genElemConnectivity (elemIndex, conn);

      Tetrahedron* tetGeom = new Tetrahedron (coordinates, conn);
      return new P13D<TetLinearTraits::NFIELDS>::Bulk (*tetGeom);
    }

  private:
    /**
     *
     * @return for each element p, populate an indexed list L of pairs (q,j), where i is index of each pair,
     *  such that
     * p shares it's i with q's edge j.
     * There should be a corresponding entry (p,i) in q's list at index j.
     */

    void getEdgeNeighborList (std::vector<std::vector<std::vector<int> > > &neighbors) const {

      unsigned int iElements = getNumElements ();

      neighbors.clear();
      neighbors.resize(iElements);

      std::vector<edgestruct> edges;

      int V1[] = { 0, 1, 0, 2, 0, 1 };
      int V2[] = { 1, 2, 2, 3, 3, 3 };

      // the 4 nodes of a tet are numbered 0..3
      // edges are 0-1, 0-2, 0-3, 1-2, 1-3, 2-3
      // the nodes corresponding to edges are picked up in an involved manner using two arrays V1 and V2
      // the edges must be sorted in a consistent manner, due to which the nodes in an edge must be
      // sorted.

      // Creating a list of all possible edges.
      for (unsigned int e = 0; e < iElements; e++) {
        neighbors[e].resize(NUM_EDGES);

        int econn[] = { connectivity[e * 4 + 0], connectivity[e * 4 + 1], connectivity[e * 4 + 2], connectivity[e * 4 + 3] };

        for (int edgenum = 0; edgenum < NUM_EDGES; edgenum++) {
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
          for (unsigned int p = 0; p < repedges.size(); p++) {
            for (unsigned int q = 0; q < repedges.size(); q++) {
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
    /* Function: SubdivideTet
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

      int sd = getSpatialDim();
      int eNodes = getNodesPerElem(); // Number of nodes per element.

      unsigned int iElements = getNumElements(); // Number of elements.
      unsigned int iNodes = getNumNodes();

      std::vector<std::vector<std::vector<int> > > neighbors;
      getEdgeNeighborList(neighbors);

      // Connectivity for mid-points of each edgeId for each element.
      int midconn[iElements][NUM_EDGES];
      int count = iNodes;

      for (unsigned int e = 0; e < iElements; e++) {
        for (int f = 0; f < NUM_EDGES; f++) {

          // Number of elements sharing edgeId 'f' of element 'e'.
          int nNeighbors = neighbors[e][f].size() / 2;

          if (nNeighbors == 0) { // Free edgeId.

            // for 0-based node numbering we increment 'count' afterwards
            // count++;
            midconn[e][f] = count;
            ++count;

          } else { // Shared edgeId
            // Find the least element neighbor number.
            int minElem = e;
            for (int p = 0; p < nNeighbors; p++) {
              if (minElem > neighbors[e][f][2 * p]) {
                minElem = neighbors[e][f][2 * p];
              }
            }

            if (int(e) == minElem) { // Allot only once for a shared edgeId.
              // for 0-based node numbering we increment 'count' afterwards
              // count++;
              midconn[e][f] = count;

              for (int p = 0; p < nNeighbors; p++) {
                int nelem = neighbors[e][f][2 * p];
                int nedge = neighbors[e][f][2 * p + 1];
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
      std::vector<double> newCoord(count * sd);
      std::vector<int> newConn;

      for (unsigned int i = 0; i < coordinates.size(); i++) {
        newCoord[i] = coordinates[i];
      }

      // Coordinates for midside nodes:
      int V1[] = { 0, 1, 0, 2, 0, 1 };
      int V2[] = { 1, 2, 2, 3, 3, 3 };
      for (unsigned int e = 0; e < iElements; e++)
        for (int f = 0; f < NUM_EDGES; f++) {
          // for 0-based node numbering, we don't need to subtract 1 from node ids in connectivity
          // int v1 = connectivity[e * eNodes + V1[f]] - 1;
          // int v2 = connectivity[e * eNodes + V2[f]] - 1;
          // for (int k = 0; k < sd; k++)
          // newCoord[(midconn[e][f] - 1) * sd + k] = 0.5 * (coordinates[v1 * sd + k] + coordinates[v2 * sd + k]);
          int v1 = connectivity[e * eNodes + V1[f]];
          int v2 = connectivity[e * eNodes + V2[f]];
          for (int k = 0; k < sd; k++) {
            newCoord[midconn[e][f] * sd + k] = 0.5 * (coordinates[v1 * sd + k] + coordinates[v2 * sd + k]);
          }
        }

      for (unsigned int e = 0; e < iElements; e++) {
        // tet 1
        int t1conn[] = { connectivity[e * eNodes + 0], midconn[e][0], midconn[e][2], midconn[e][4] };
        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t1conn[i]);
        }

        // tet 2
        int t2conn[] = { midconn[e][0], connectivity[e * eNodes + 1], midconn[e][1], midconn[e][5] };
        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t2conn[i]);
        }

        // tet3
        int t3conn[] = { midconn[e][2], midconn[e][1], connectivity[e * eNodes + 2], midconn[e][3] };
        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t3conn[i]);
        }

        // tet4
        int t4conn[] = { midconn[e][4], midconn[e][5], midconn[e][3], connectivity[e * eNodes + 3] };

        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t4conn[i]);
        }

        // tet5
        int t5conn[] = { midconn[e][0], midconn[e][3], midconn[e][4], midconn[e][5] };
        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t5conn[i]);
        }

        // tet6
        int t6conn[] = { midconn[e][0], midconn[e][3], midconn[e][5], midconn[e][1] };
        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t6conn[i]);
        }

        // tet 7
        int t7conn[] = { midconn[e][0], midconn[e][3], midconn[e][1], midconn[e][2] };
        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t7conn[i]);
        }

        // tet 8
        int t8conn[] = { midconn[e][0], midconn[e][3], midconn[e][2], midconn[e][4] };
        for (int i = 0; i < eNodes; i++) {
          newConn.push_back(t8conn[i]);
        }
      }
      coordinates.clear();
      connectivity.clear();
      coordinates.assign(newCoord.begin(), newCoord.end());
      connectivity.assign(newConn.begin(), newConn.end());

      // nodes = int(coordinates.size() / 3);
      // elements = int(connectivity.size() / 4);
    }
};

#endif // _TET_LINEAR_COORD_CONN_H_
