/*
 * CoordConn.h
 *
 *  Created on: Jun 16, 2011
 *      Author: amber
 */

#ifndef COORDCONN_H_
#define COORDCONN_H_

#include "AuxDefs.h"
#include "Element.h"
#include "P12DElement.h"
#include "P13DElement.h"
#include "ElementGeometry.h"
#include "Triangle.h"
#include "Tetrahedron.h"
#include "Femap.h"

#include <cassert>

#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>

class CoordConn {


public:
  CoordConn () {}

  virtual ~CoordConn () {}

  virtual const std::vector<GlobalNodalIndex>& getConnectivity () const = 0;

  virtual const std::vector<double>& getCoordinates () const = 0;

  virtual size_t getSpatialDim () const = 0;

  virtual size_t getNodesPerElem () const = 0;

  virtual size_t getTopology () const = 0;

  virtual void subdivide () = 0;

  virtual void initFromFileData (const FemapInput& neu) = 0;

  virtual size_t getNumNodes () const = 0;

  virtual size_t getNumElements () const = 0;

  virtual Element* makeElem (const size_t elemIndex) const = 0;

protected:
  virtual void genElemConnectivity (size_t elemIndex, std::vector<GlobalNodalIndex>& elemConn) const = 0;


};

template <size_t SPD, size_t NODES_PER_ELEM, size_t TOPO>
class AbstractCoordConn: public CoordConn {
protected:
  std::vector<GlobalNodalIndex> connectivity;
  std::vector<double> coordinates;

public:
  AbstractCoordConn (): CoordConn() {

  }

  AbstractCoordConn (const AbstractCoordConn& that):
    CoordConn(that), connectivity (that.connectivity), coordinates (that.coordinates) {

  }

  AbstractCoordConn& operator = (const AbstractCoordConn& that) {
    CoordConn::operator = (that);
    if (this != &that) {
      connectivity = that.connectivity;
      coordinates = that.coordinates;
    }
    return (*this);
  }

  virtual inline size_t getSpatialDim () const {
    return SPD;
  }

  virtual inline size_t getNodesPerElem () const {
    return NODES_PER_ELEM;
  }

  virtual inline size_t getTopology () const {
    return TOPO;
  }

  virtual const std::vector<GlobalNodalIndex>& getConnectivity () const {
    return connectivity;
  }

  virtual const std::vector<double>& getCoordinates () const {
    return coordinates;
  }

  virtual size_t getNumNodes () const {
    return getCoordinates ().size () / getSpatialDim ();
  }

  virtual size_t getNumElements () const {
    return getConnectivity ().size () / getNodesPerElem ();
  }

  virtual void initFromFileData (const FemapInput& neu) {

    size_t nodes = neu.getNumNodes ();
    size_t elements = neu.getNumElements (getTopology ());

    coordinates.clear ();
    coordinates.resize (nodes * getSpatialDim ());

    connectivity.clear ();
    connectivity.resize (elements * getNodesPerElem ());

    transferNodes (neu);
    transferElements (neu);
  }


protected:
  virtual void genElemConnectivity (size_t elemIndex, std::vector<GlobalNodalIndex>& conn) const {
    const size_t npe = getNodesPerElem ();
    conn.clear();

    for (size_t i = 0; i < npe; ++i) {
      conn.push_back (connectivity[elemIndex * npe + i]);
    }
  }

private:
  void transferNodes (const FemapInput& neu) {

    size_t n, d;
    for (n = 0; n < neu.getNumNodes (); n++) {
      const femapNode& nd = neu.getNode (n);
      for (d = 0; d < getSpatialDim (); d++) {
        coordinates[getSpatialDim () * n + d] = nd.x[d];
      }
    }

  }


  void transferElements (const FemapInput& neu) {
    size_t i, j;
    for (i = 0; i < neu.getNumElements (); i++) {
      const femapElement& e = neu.getElement (i);
      if (e.topology == getTopology ()) {
        for (j = 0; j < getNodesPerElem (); ++j) {
          // changed following to make node ids start from 0.
          // connectivity[nodesPerElem * iv + j] = neu.getNodeId(e.node[j]) + 1;
          connectivity[getNodesPerElem () * i + j] = neu.getNodeId (e.node[j]);
        }

      } else {
        std::ostringstream ss;
        ss << "Warning: topology of element " << neu.getElementId (e.id)
          << " is not supported for conversion to ADLIB.  Skipping. " << std::endl;
        throw std::runtime_error (ss.str ());
      }
    }

    return;
  }


};



struct edgestruct {
  size_t elemId;
  size_t edgeId;
  GlobalNodalIndex node0;
  GlobalNodalIndex node1;

  edgestruct(size_t ielem, size_t iedge, GlobalNodalIndex _node0, GlobalNodalIndex _node1) :
    elemId(ielem), edgeId(iedge) {

    // sort the args in the increasing order
    // can't sort fields due to const
    if (_node1 < _node0) {
      std::swap (_node0, _node1);
    }

    // sort of node id's of an edge in a consistent manner is necessary
    // in order to sort a list of edgestruct objects 

    node0 = _node0;
    node1 = _node1;

    assert (node0 <= node1);

  }


  bool operator < (const edgestruct &that) const {
    // compare the nodes of the two edges
    int result = compare (that);
    return result < 0;
  }

  inline int compare (const edgestruct& that) const {
    int result = this->node0 - that.node0;
    if (result == 0) {
      result = this->node1 - that.node1;
    }
    return result;
  }

};

#endif /* COORDCONN_H_ */
