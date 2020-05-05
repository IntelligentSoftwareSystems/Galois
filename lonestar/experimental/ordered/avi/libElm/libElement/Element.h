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
 * Element.h: Basic Element over which fields are defined.
 * It contains an element geometry, shape functions etc.
 *
 * DG++
 *
 * Created by Adrian Lew on 9/2/06.
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

#ifndef ELEMENT
#define ELEMENT

#include <vector>
#include "AuxDefs.h"
#include "ElementGeometry.h"
#include "BasisFunctions.h"

// TODO: add P1nDBoundaryTrace and P1nDMap

/**
   \brief Element: abstract base class for any element

   An Element is a convex polytope with possibly-multiple discrete
   functional spaces, one for each field, with support on it.

   An element has:\n
          1) A geometry. The connectivity of vertices that define the convex
   hull.\n 2) A group of scalar functional spaces. Each functional space is
   defined by a set of basis functions with support on the element. Each
   functional space has an associated number of degrees of freedom to it: the
   components of any function in the space in the chosen basis.\n

   A field is a scalar-valued function defined over the element.\n
   Each field may have a different underlying functional space.\n
   Each field may have a different quadrature rule. Different quadrature rules
   will also be handled with different elements, or through inheritance. The
   consistency of the quadrature rule when different fields are integrated
   together is in principle not checked by the element hierarchy. \n Clearly,
   isoparametric elements will not necessarily be convex, but Element class can
   still be used.

   As a convention, fields are numbered starting from 0.

   \todo It would be nice to have vectors and tensors as fields, where for each
   component one assigns a single set of shape functions and quadrature points.
   The current abstraction is flexible in the sense that it does not enforce
   vector or tensor fields to have the same quadrature and shapes.

   \todo Need to explain what the expected order of the vectors getShapes() and
   getDShapes() is.

   \todo Need to explain that either getShapes or getDShapes may return an empty
   vector, signaling that those values are not available.
*/

class Element {
public:
  inline Element() {}
  inline virtual ~Element() {}
  inline Element(const Element&) {}
  virtual Element* clone() const = 0;

  // Accessors/Mutators:

  //! Number of different fields
  virtual size_t getNumFields() const = 0;

  //! Number of degrees of freedom of one of the fields.
  virtual size_t getDof(size_t field) const = 0;

  //! Number of derivatives per shape function for one of the fields
  virtual size_t getNumDerivatives(size_t field) const = 0;

  //!  Shape functions at quadrature points of one of the fields
  virtual const VecDouble& getShapes(size_t field) const = 0;

  //!  Shape function derivatives at quadrature points of one of the fields
  virtual const VecDouble& getDShapes(size_t field) const = 0;

  //! Integration weights of a given field
  virtual const VecDouble& getIntegrationWeights(size_t field) const = 0;

  //! Integration point coordinates of a given field
  virtual const VecDouble& getIntegrationPtCoords(size_t field) const = 0;

  //! Value of shape function "shapenumber" of field "field"
  //! at quadrature point "quad"
  virtual double getShape(size_t field, size_t quad,
                          size_t shapenumber) const = 0;

  //! Value of derivative of shape function "shapenumber" of field "field"
  //! at quadrature point "quad" in direction "dir"
  virtual double getDShape(size_t field, size_t quad, size_t shapenumber,
                           size_t dir) const = 0;

  //! Access to Element Geometry
  virtual const ElementGeometry& getGeometry() const = 0;
};

/**
   \brief    Element_: Abstract implementation of an Element class

   Element_ constructs a finite element type. The ElementGeometry is
   specified in derived classes. The idea is to use this class to derive
   different finite element types.


   Element_ is defined by:\n
   1)  Access to a (ElementGeometry &)EG that defines the geometry of
   the element through a map a reference domain \f$\hat{\Omega}\f$ to
   the real element shape (or an approximation, such as in
   isoparametric elements)\n
   2) An array (BasisFunctions *) LocalShapes all constructed over EG.
   3) A virtual function to be defined by derived classses getFieldShapes. The
   value of LocalShapes[getFieldShapes(FieldNumber)] returns the shape
   functions of field FieldNumber.  This map is needed since the same
   BasisFunctions object can be used for several fields.

*/

class Element_ : public Element {
private:
  void copy(const Element_& that) {
    for (size_t i = 0; i < that.LocalShapes.size(); i++) {
      LocalShapes.push_back((that.LocalShapes[i])->clone());
    }
  }

  void destroy() {
    for (size_t i = 0; i < LocalShapes.size(); i++) {
      delete LocalShapes[i];
      LocalShapes[i] = NULL;
    }
  }

public:
  inline Element_() : Element() {}

  inline virtual ~Element_() { destroy(); }

  Element_(const Element_& OldElement_) : Element(OldElement_) {
    copy(OldElement_);
  }

  Element_& operator=(const Element_& that) {
    if (this != &that) {
      destroy();
      copy(that);
    }
    return (*this);
  }

  virtual Element_* clone() const = 0;

  inline size_t getDof(size_t field) const {
    return LocalShapes[getFieldShapes(field)]->getBasisDimension();
  }
  inline size_t getNumDerivatives(size_t field) const {
    return LocalShapes[getFieldShapes(field)]
        ->getNumberOfDerivativesPerFunction();
  }
  inline const VecDouble& getShapes(size_t field) const {
    return LocalShapes[getFieldShapes(field)]->getShapes();
  }
  inline const VecDouble& getDShapes(size_t field) const {
    return LocalShapes[getFieldShapes(field)]->getDShapes();
  }
  inline const VecDouble& getIntegrationWeights(size_t field) const {
    return LocalShapes[getFieldShapes(field)]->getIntegrationWeights();
  }
  inline const VecDouble& getIntegrationPtCoords(size_t field) const {
    return LocalShapes[getFieldShapes(field)]->getQuadraturePointCoordinates();
  }

  inline double getShape(size_t field, size_t quad, size_t shapenumber) const {
    return getShapes(field)[quad * getDof(field) + shapenumber];
  }

  inline double getDShape(size_t field, size_t quad, size_t shapenumber,
                          size_t dir) const {
    return getDShapes(field)[quad * getDof(field) * getNumDerivatives(field) +
                             shapenumber * getNumDerivatives(field) + dir];
  }

protected:
  //! addBasisFunctions adds a BasisFunctions pointer at the end of LocalShapes
  //! The i-th added  BasisFunctions pointer is referenced when getFieldShapes
  //! returns  the integer i-1
  inline void addBasisFunctions(const BasisFunctions& BasisFunctionsPointer) {
    LocalShapes.push_back(BasisFunctionsPointer.clone());
    // XXX: amber: clone has no effect (see ShapesEvaluated__ copy constructor)
    // and it seems that clone is not needed LocalShapes.push_back(
    // (const_cast<BasisFunctions*> (&BasisFunctionsPointer)) );
  }

  //! getFieldShapes returns the position in LocalShapes in which
  //! the shape functions for field Field are.
  virtual size_t getFieldShapes(size_t Field) const = 0;

  //! returns the length of LocalShapes
  inline size_t getNumShapes() const { return LocalShapes.size(); }

private:
  std::vector<BasisFunctions*> LocalShapes;
};

/**
 \brief
 SpecificElementFamily: classes that contain all Element types that form
 a family. For example, bulk and boundary interpolation.
 */

class SpecificElementFamily {};

/**
 \brief LocalToGlobalMap class: map the local degrees of freedom
 of each Element to the global ones.

 The Local to Global map is strongly dependent on how the program
 that utilizes the Element objects is organized. The objective of
 this class is then to define the interface that the derived
 objects should have.

 There will generally not be a single LocalToGlobalMap object per
 Element, but rather only one LocalToGlobalMap for all of
 them. Hence, the interface requires a way to specify which
 element is being mapped.

 Convention:\n
 Fields and Dofs start to be numbered at 0
 */
class LocalToGlobalMap {
public:
  inline LocalToGlobalMap() {}
  inline virtual ~LocalToGlobalMap() {}
  inline LocalToGlobalMap(const LocalToGlobalMap&) {}
  virtual LocalToGlobalMap* clone() const = 0;

  // //! @param ElementMapped: GlobalElementIndex of the Element to be mapped.
  // //! This sets
  // //! the default Element  object whose field degrees of freedom are
  // //! mapped.
  // // XXX: commented out (amber)
  // // virtual void set (const GlobalElementIndex &ElementMapped) = 0;

  // //! @param field field number in the element, 0\f$ \le \f$ field \f$\le\f$
  // Nfields-1\n
  // //! @param dof   number of degree of freedom in that field,
  // //! \f$ 0 \le \f$ dof \f$\le\f$ Ndofs-1 \n
  // //! map returns the GlobalDofIndex associated to degree of freedom "dof" of
  // field "field"
  // //! in the default Element object. The latter is
  // //! set with the function set(const Element &).
  // // XXX: commented out (amber)
  // // virtual const GlobalDofIndex map (size_t field, size_t dof) const = 0;

  //! @param field field number in the element, 0\f$  \le \f$ field \f$\le\f$
  //! Nfields-1\n
  //! @param dof   number of degree of freedom in that field, \f$ 0 \le \f$ dof
  //! \f$\le\f$ Ndofs-1 \n
  //! @param ElementMapped: GlobalElementIndex of the Element whose degrees
  //! of freedom are being mapped\n
  //! map returns the GlobalDofIndex associated to degree of freedom "dof"
  //! of field "field"
  //! in element MappedElement.
  virtual GlobalDofIndex map(size_t field, size_t dof,
                             const GlobalElementIndex& ElementMapped) const = 0;

  //! Total number of elements that can be mapped. Usually, total number of
  //! elements in the mesh.
  virtual size_t getNumElements() const = 0;

  //! Number of fields in an element mapped
  virtual size_t
  getNumFields(const GlobalElementIndex& ElementMapped) const = 0;

  //! Number of dofs in an element mapped in a given field
  virtual size_t getNumDof(const GlobalElementIndex& ElementMapped,
                           size_t field) const = 0;

  //! Total number of dof in the entire map
  virtual size_t getTotalNumDof() const = 0;
};

#endif
