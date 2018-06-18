/**
 * ElementGeometry.h: Geometry of an element. e.g. a triangle or tetrahedron
 * DG++
 *
 * Created by Adrian Lew on 9/4/06.
 *
 * Copyright (c) 2006 Adrian Lew
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef ELEMENTGEOMETRY
#define ELEMENTGEOMETRY

#include "AuxDefs.h"
#include <string>
#include <vector>
#include <iostream>

#include <cmath>
/**
   \brief ElementGeometry: Defines the geometry of the polytope over which the
   interpolation takes place

   ElementGeometry consists of:\n
   1) A set of vertices that define a convex hull, the domain of the polytope\n
   2) A map from a parametric, reference polytope to the real
   polytope. This map is one-to-one, and may have domain and range
   in Euclidean spaces of different dimensions. In this way it is
   possible to map the parametric configuration of a planar
   triangle into three-dimensional real space as needed for plates
   and shells.\n
   3) A name for the polytope, for identification purposes whenever needed.\n

   The idea of the class is to avoid having copies of the vertices coordinates
   in the object, only the connectivity.
*/

class ElementGeometry {
public:
  inline ElementGeometry() {}

  inline virtual ~ElementGeometry() {}

  inline ElementGeometry(const ElementGeometry&) {}

  virtual ElementGeometry* clone() const = 0;

  //! @return number of vertices
  virtual size_t getNumVertices() const = 0;

  //!@return  ref to Vertices of the polytope.
  virtual const VecSize_t& getConnectivity() const = 0;

  //! @return Name of type of polytope.
  virtual const std::string getPolytopeName() const = 0;

  //! @return spatial dimension e.g. 2 for 2D
  virtual size_t getSpatialDimension() const = 0;

  //! Number of dimensions in parametric configuration
  virtual size_t getParametricDimension() const = 0;

  //! Number of dimensions in the real configuration
  virtual size_t getEmbeddingDimension() const = 0;

  //! map from parametric to real configuration
  //! @param X parametric coordinates
  //! @param Y returned real coordinates
  virtual void map(const double* X, double* Y) const = 0;

  //! Derivative of map from parametric to real configuration
  //! @param X parametric coordinates.
  //! @param Jac returns absolute value of the Jacobian of the map.
  //! @param DY  returns derivative of the map.
  //! Here DY[a*getEmbeddingDimension()+i]
  //! contains the derivative in the a-th direction
  //! of the i-th coordinate.
  virtual void dMap(const double* X, double* DY, double& Jac) const = 0;

  //! Consistency test for map and its derivative
  //! @param X parametric coordinates at which to test
  //! @param Pert size of the perturbation with which to compute numerical
  //! derivatives (X->X+Pert)
  virtual bool consistencyTest(const double* X, const double Pert) const = 0;

  //! Number of faces the polytope has
  virtual size_t getNumFaces() const = 0;

  //! Creates and returns a new ElementGeometry object corresponding
  //! to face "e" in the polytope. The object has to be destroyed
  //! with delete by the recipient.
  //!
  //! @param e face number, starting from 0
  //!
  //! Returns a null pointer if "e" is out of range
  virtual ElementGeometry* getFaceGeometry(size_t e) const = 0;

  //! Computes the Inner radius of the ElementGeometry object
  //!
  //! This is defined as the radius of the largest sphere that can be fit inside
  //! the polytope.
  virtual double getInRadius() const = 0;

  //! Computes the Outer radius of the ElementGeometry object
  //!
  //! This is defined as the radius of the smallest sphere that contains the
  //! object.
  virtual double getOutRadius() const = 0;

  //! Compute external normal for a face
  //!
  //! @param e: face number for which the normal is desired
  //! @param vNormal: output of the three Cartesian components of the normal
  //! vector
  virtual void computeNormal(size_t e, VecDouble& vNormal) const = 0;

  /**
   * Returns the value of dimension 'i' of local node 'a' of the eleement
   *
   * @param a local index of the node in [0..numNodes)
   * @param i local index of dimension (x or y or z) in [0..Dim)
   */
  virtual double getCoordinate(size_t a, size_t i) const = 0;

  /**
   * Computes the center of the element (the way center is defined may be
   * different for different elements)
   *
   * @param center output vector containing the coordinates of the center
   */
  virtual void computeCenter(VecDouble& center) const = 0;
};

/**
 * Base class with common functionality
 */
template <size_t SPD>
class AbstractGeom : public ElementGeometry {
private:
  const VecDouble& globalCoordVec;
  VecSize_t connectivity;

protected:
  static const size_t SP_DIM = SPD;
  /**
   * @return ref to the vector that contains global coordinates for all mesh
   * nodes
   */
  const VecDouble& getGlobalCoordVec() const { return globalCoordVec; }

public:
  /**
   * @param globalCoordVec is a reference to the vector containing coordinates
   * of all nodes Coordinates of node i in N dimensional space are in locations
   * [N*i, N*(i+1))
   * @param connectivity is a vector containing ids of nodes of this element in
   * the mesh
   */

  AbstractGeom(const VecDouble& globalCoordVec, const VecSize_t& connectivity)
      : ElementGeometry(), globalCoordVec(globalCoordVec),
        connectivity(connectivity) {}

  AbstractGeom(const AbstractGeom<SPD>& that)
      : ElementGeometry(that), globalCoordVec(that.globalCoordVec),
        connectivity(that.connectivity) {}

  virtual size_t getSpatialDimension() const { return SP_DIM; }

  virtual const VecSize_t& getConnectivity() const { return connectivity; }

  virtual bool consistencyTest(const double* X, const double Pert) const {
    double* DYNum =
        new double[getParametricDimension() * getEmbeddingDimension()];
    double* DY = new double[getParametricDimension() * getEmbeddingDimension()];
    double* Xpert  = new double[getParametricDimension()];
    double* Yplus  = new double[getEmbeddingDimension()];
    double* Yminus = new double[getEmbeddingDimension()];
    double Jac;

    if (Pert <= 0)
      std::cerr << "ElementGeometry::ConsistencyTest - Pert cannot be less or "
                   "equal than zero\n";

    for (size_t a = 0; a < getParametricDimension(); a++)
      Xpert[a] = X[a];

    dMap(X, DY, Jac);

    for (size_t a = 0; a < getParametricDimension(); a++) {
      Xpert[a] = X[a] + Pert;
      map(Xpert, Yplus);

      Xpert[a] = X[a] - Pert;
      map(Xpert, Yminus);

      Xpert[a] = X[a];

      for (size_t i = 0; i < getEmbeddingDimension(); i++)
        DYNum[a * getEmbeddingDimension() + i] =
            (Yplus[i] - Yminus[i]) / (2 * Pert);
    }

    double error     = 0;
    double normX     = 0;
    double normDYNum = 0;
    double normDY    = 0;

    for (size_t a = 0; a < getParametricDimension(); a++) {
      normX += X[a] * X[a];

      for (size_t i = 0; i < getEmbeddingDimension(); i++) {
        error += pow(DY[a * getEmbeddingDimension() + i] -
                         DYNum[a * getEmbeddingDimension() + i],
                     2.);
        normDY += pow(DY[a * getEmbeddingDimension() + i], 2.);
        normDYNum += pow(DYNum[a * getEmbeddingDimension() + i], 2.);
      }
    }
    error     = sqrt(error);
    normX     = sqrt(normX);
    normDY    = sqrt(normDY);
    normDYNum = sqrt(normDYNum);

    delete[] Yplus;
    delete[] Yminus;
    delete[] Xpert;
    delete[] DYNum;
    delete[] DY;

    if (error * (normX + Pert) <
        (normDY < normDYNum ? normDYNum : normDY) * Pert * 10)
      return true;
    else
      return false;
  }

  /**
   * @param a node id
   * @param i dimension id
   * @return value of dimension 'i' of coordinates of node 'a'
   */
  virtual double getCoordinate(size_t a, size_t i) const {
    // 0-based numbering of nodes in the mesh
    size_t index = getConnectivity()[a] * getSpatialDimension() + i;
    return globalCoordVec[index];
  }

  virtual void computeCenter(VecDouble& center) const {
    std::cerr << "computeCenter not implemented" << std::endl;
    abort();
  }
};

#endif
