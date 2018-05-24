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

/**
 * This class maintains connectivity and coordinates of the mesh read from a file.
 * Connectivity is the ids of the nodes of each element, where nodes are numbered from * 0..numNodes-1
 * Coordinates is the 2D or 3D coordinates of each node in the mesh
 * The elements themselves have ids 0..numElements-1
 */
class CoordConn {


public:
  CoordConn () {}

  virtual ~CoordConn () {}

  /**
   * Connectivity of all elements in a single vector. Let NPE = nodes per element, then connectivity of 
   * element i is in the range [NPE*i, NPE*(i+1))
   *
   * @return ref to vector
   */
  virtual const VecSize_t& getConnectivity () const = 0;

  /** 
   * Coordinates of all nodes in the mesh in a single vector. Let SPD = number of spatial dimensions
   * e.g. 2D or 3D, the coordinates for node i are in the range [i*SPD, (i+1)*SPD)
   *
   * @return ref to vector
   */
  virtual const VecDouble& getCoordinates () const = 0;

  virtual size_t getSpatialDim () const = 0;

  virtual size_t getNodesPerElem () const = 0;

  /**
   * specific to file input format
   */
  virtual size_t getTopology () const = 0;

  /**
   * subdivide each element into smaller elements
   */
  virtual void subdivide () = 0;

  virtual void initFromFileData (const FemapInput& neu) = 0;

  virtual size_t getNumNodes () const = 0;

  virtual size_t getNumElements () const = 0;

  /**
   * helper for MeshInit
   * The derived class decides the kind of element and element geometry to 
   * instantiate for each element addressed by elemIndex
   *
   * @param elemIndex
   * @return Element*
   */
  virtual Element* makeElem (const size_t elemIndex) const = 0;

protected:

  /**
   * populates vector elemConn with 
   * the connectivity of element indexed by elemIndex
   * @see CoordConn::getConnectivity()
   *
   * @param elemIndex
   * @param elemConn
   *
   */
  virtual void genElemConnectivity (size_t elemIndex, VecSize_t& elemConn) const = 0;


};

/**
 *
 * Common functionality and data structures
 */
template <size_t SPD, size_t NODES_PER_ELEM, size_t TOPO>
class AbstractCoordConn: public CoordConn {
protected:
  VecSize_t connectivity;
  VecDouble coordinates;

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

  virtual const VecSize_t& getConnectivity () const {
    return connectivity;
  }

  virtual const VecDouble& getCoordinates () const {
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
  virtual void genElemConnectivity (size_t elemIndex, VecSize_t& conn) const {
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
        std::cerr << "Warning: topology " << e.topology << " of element " << neu.getElementId (e.id)
          << " is not supported for conversion to ADLIB.  Skipping. " << std::endl;
        abort ();
      }
    }

    return;
  }


};



/**
 * represents an edge between two mesh nodes
 */
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


  /**
   * ordering based on node ids
   *
   * @param that
   */
  bool operator < (const edgestruct &that) const {
    // compare the nodes of the two edges
    int result = compare (that);
    return result < 0;
  }

  /**
   * comparison function that compares
   * two objects based on the node ids in the edge
   * Therefore it's necessary to store the node ids within an edge
   * in sorted order to allow lexicographic comparison
   *
   * @param that
   */
  inline int compare (const edgestruct& that) const {
    int result = this->node0 - that.node0;
    if (result == 0) {
      result = this->node1 - that.node1;
    }
    return result;
  }

};

#endif /* COORDCONN_H_ */
