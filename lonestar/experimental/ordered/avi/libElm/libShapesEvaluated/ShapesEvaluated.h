/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

/**
 * ShapesEvaluated.h
 * DG++
 *
 * Created by Adrian Lew on 9/7/06.
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

#ifndef SHAPESEVALUATED
#define SHAPESEVALUATED

#include <string>

#include "Shape.h"
#include "Quadrature.h"
#include "ElementGeometry.h"
#include "BasisFunctions.h"
#include "Linear.h"

/**
   ShapesEvaluated: Evaluation of the shape functions and derivatives at
   the quadrature points. Abstract class.

   ShapesEvaluated is the class that takes the element geometry, a
   quadrature rule and shape functions on the reference element, and
   computes the values of the shape functions and their derivatives at
   the quadrature points. As opposed to Shapes, there will be one or
   more ShapesEvaluated objects per element in the mesh. Each different
   interpolation needed in an element will have a different
   ShapesEvaluated object.


   This class provides all functionality for the templated derived classes
   ShapesEvaluated__. Only two abstract functions are left to be defined by
   derived classes, accessShape and accessQuadrature.

   Objects in this class should only be constructed by derived classes.

   ShapesEvaluated evaluates ElementGeometry::(D)map at the quadrature
   point (map)Coordinates using Quadrature::getQuadraturePoint().

   ShapesEvaluated evaluates Shape functions at the quadrature
   point (Shape)Coordinates using Quadrature::getQuadraturePointShape().

   \warning: The type of coordinates used to evaluate functions in
   Shape and ElementGeometry::(D)map (barycentric, Cartesian, etc.)
   should be consistent with those provided in
   Quadrature::getQuadraturePoint.  In other words, if the Quadrature
   object returns Cartesian coordinates, the Shape and
   ElementGeometry objects should evaluate functions taking
   Cartesian coordinates as arguments.

   \todo It would be nice to provide an interface of iterators to navigate
   the vectors, so that it is not necessary to remember in which order the
   data is stored in the vector. In the interest of simiplicity, this is
   for the moment skipped.

   \todo It would useful to have the option of not computing the derivatives of
   the shape functions if not needed. Right now, it is not possible.


   \todo Make coordinate types so that it is not necessary to check
   whether one is using the right convention between the three related
   classes.

   \todo The computation of derivatives of shape functions with
   respect to the coordinates of the embedding space can only be
   performed if the ElementGeometry::map has domain and ranges of the
   same dimension. Otherwise, the derivatives should be computed with
   respect to some system of coordinates on the manifold. This will
   likely need to be revisited in the case of shells.

   \todo Need to fix the Lapack interface... it is too particular to
   Mac here, and even not the best way to do it in Mac...
*/

class ShapesEvaluated : public BasisFunctions {
protected:
  ShapesEvaluated() {}
  inline virtual ~ShapesEvaluated() {}
  ShapesEvaluated(const ShapesEvaluated& SEI);

public:
  // Accessors/Mutators:
  inline const VecDouble& getShapes() const { return LocalShapes; }

  inline const VecDouble& getDShapes() const { return LocalDShapes; }

  inline const VecDouble& getIntegrationWeights() const { return LocalWeights; }

  inline const VecDouble& getQuadraturePointCoordinates() const {
    return LocalCoordinates;
  }

  //! returns the number of shape functions provided
  inline size_t getBasisDimension() const {
    return accessShape().getNumFunctions();
  }

  //! returns the number of directional derivative for each shape function
  inline size_t getNumberOfDerivativesPerFunction() const {
    return LocalDShapes.size() / LocalShapes.size();
  }

  //! returns the number of  number of coordinates for each Gauss point
  inline size_t getSpatialDimensions() const {
    return LocalCoordinates.size() / LocalWeights.size();
  }

protected:
  //! Returns the specific Shape objects in derived classes.
  virtual const Shape& accessShape() const = 0;

  //! Quadrature type
  //! Returns the specific  Quadrature objects in derived classes.
  virtual const Quadrature& accessQuadrature() const = 0;

  //! Since it is not possible to have a virtual constructor,
  //! one is emulated below only accessible from derived classes
  //! The virtual aspect are the calls to accessShape and accessQuadrature.
  void createObject(const ElementGeometry& eg);

private:
  VecDouble LocalShapes;
  VecDouble LocalDShapes;
  VecDouble LocalWeights;
  VecDouble LocalCoordinates;
};

/**
   \brief ShapesEvaluated__: This class is the one that brings the
   flexibility for building shape functions of different types
   evaluated at different quadrature points.

   The class takes the Shape and Quadrature types as template arguments.
   An object is constructed by providing an ElementGeometry object.

   ShapesEvaluated_ and ShapesEvaluated__ could have been made into a single
   templated class. By splitting them the templated part of the class is very
   small.

   \todo Should I perhaps have the constructors of the class to be
   protected, making the classes using ShapesEvaluated__ friends?
   Since the ElementGeometry is not stored or referenced from within
   the class, it would prevent unwanted mistakes.

   \warning When a ShapesEvaluated__ object is constructed on a
   geometry where the parametric and embedding dimensions are
   different, the constructor is not warning that it does not compute
   the derivatives for the shape functions.  These objects, however,
   still provide the values of the Shape functions themselves.

   \todo In the future, as remarked in ShapesEvaluated as well, we need to
   separate the need to provide the Shape function values from the one of
   providing the derivatives as well, perhaps through multiple inheritance.
*/

template <const Shape* const& ShapeObj, const Quadrature* const& QuadObj>
class ShapesEvaluated__ : public ShapesEvaluated {
public:
  inline ShapesEvaluated__(const ElementGeometry& EG) : ShapesEvaluated() {
    createObject(EG);
  }

  virtual ShapesEvaluated__* clone() const {
    return new ShapesEvaluated__(*this);
  }

  ShapesEvaluated__(const ShapesEvaluated__& SE) : ShapesEvaluated(SE) {}

  const Shape& accessShape() const { return *ShapeObj; }
  const Quadrature& accessQuadrature() const { return *QuadObj; }
};

// Build specific ShapesEvaluated

//! Specific ShapesEvaluated types
class SpecificShapesEvaluated {};

//! Shape functions for P12D elements: Linear functions on Triangles\n
//! It contains two types of traces\n
//! 1) ShapesP12D::Faces are linear functions on Segments, and their degrees of
//! freedom are those associated to the nodes of the Segment.
//!
//! 2) ShapesP12D::FaceOne, ShapesP12D::FaceTwo, ShapesP12D::FaceThree are the
//! full set of linear shape functions in the element evaluated at quadrature
//! points in each one of the faces. Instead of having then 2 degrees of freedom
//! per field per face, there are here 3 degrees of freedom per field per face.
//! Of course, some of these values are trivially zero, i.e., those of the shape
//! functions associated to the node opposite to the face where the quadrature
//! point is.  These are provided since  ocassionally it may be necessary to
//! have the boundary fields have the same number of degrees of freedom as the
//! bulk fields. In the most general case of arbitrary bases,  there is
//! generally no shape functions that is identically zero on the face, and hence
//! bulk and trace fields have the same number of degrees of freedom. With these
//! shape functions, for example, it is possible to compute
//!  the normal derivative of each basis function at the boundary.

class ShapesP12D : public SpecificShapesEvaluated {
public:
  //!  Shape functions on reference triangle
  static const Shape* const P12D;

  //!  Shape functions on reference segment
  static const Shape* const P11D;

  //! Bulk shape functions
  typedef ShapesEvaluated__<P12D, Triangle_1::Bulk> Bulk;

  //! Shape functions for FaceOne, FaceTwo and FaceThree
  typedef ShapesEvaluated__<P11D, Line_1::Bulk> Faces;

  //! Full shape functions for FaceOne
  typedef ShapesEvaluated__<P12D, Triangle_1::FaceOne> FaceOne;
  //! Full shape functions for FaceTwo
  typedef ShapesEvaluated__<P12D, Triangle_1::FaceTwo> FaceTwo;
  //! Full shape functions for FaceThree
  typedef ShapesEvaluated__<P12D, Triangle_1::FaceThree> FaceThree;
};

//! Shape functions for P11D elements: Linear functions on segments
class ShapesP11D : public SpecificShapesEvaluated {
public:
  //!  Shape functions on reference segment
  static const Shape* const P11D;

  //! Bulk shape functions
  typedef ShapesEvaluated__<P11D, Line_1::Bulk> Bulk;
};

#endif
