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
 * P1nDElement.h: Common base class for 2D/3D elements with linear shape
 * functions
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

#ifndef P1NDELEMENT_H_
#define P1NDELEMENT_H_

#include <vector>
#include <cassert>

#include "AuxDefs.h"
#include "Element.h"
#include "ElementBoundaryTrace.h"
#include "ElementGeometry.h"

template <size_t NF>
class P1nDElement : public Element_ {
private:
  const ElementGeometry& elemGeom;

public:
  //! constructor
  //! @param _elemGeom element geometry
  P1nDElement(const ElementGeometry& _elemGeom)
      : Element_(), elemGeom(_elemGeom) {}

  //! copy constructor
  //! @param that
  P1nDElement(const P1nDElement& that)
      : Element_(that), elemGeom(that.elemGeom) {}

  //! @see Element::getNumFields
  virtual size_t getNumFields() const { return NF; }

  //! @see Element::getGeometry
  virtual const ElementGeometry& getGeometry() const { return elemGeom; }

protected:
  //! @see Element_::getFieldShapes
  size_t getFieldShapes(size_t field) const { return 0; }
};

/**
 * Common base class for 2D/3D linear traces
 */
template <size_t NF>
class P1nDTrace : public P1nDElement<NF> {
public:
  //! Range of FaceIndices available to enumerate faces\n
  //! When providing a FaceLabel as an argument there is automatic
  //! control of its range
  enum FaceLabel { FaceOne = 0, FaceTwo = 1, FaceThree = 2, FaceFour = 3 };

  //! TwoDofs indicates Segment<2> boundary elements, with two dofs per field \n
  //! ThreeDofs indicates Triangle<2> boundary elements, with three dofs per
  //! field. The shape functions in P12DElement are just evaluated at quadrature
  //! points on each face\n FourDofs is a Tetrahedron
  enum ShapeType { TwoDofs, ThreeDofs, FourDofs };

  P1nDTrace(const P1nDElement<NF>& baseElem) : P1nDElement<NF>(baseElem) {}

  P1nDTrace(const P1nDTrace<NF>& that) : P1nDElement<NF>(that) {}
};

/**
 * Common base class for boundary traces
 * @see ElementBoundaryTraces
 */

template <size_t NF>
class P1nDBoundaryTraces : public ElementBoundaryTraces_ {
private:
  MatDouble normals;

public:
  typedef typename P1nDTrace<NF>::FaceLabel FaceLabel;
  typedef typename P1nDTrace<NF>::ShapeType ShapeType;

  P1nDBoundaryTraces(const P1nDElement<NF>& baseElem,
                     const std::vector<FaceLabel>& faceLabels,
                     const ShapeType& shapeType)
      : ElementBoundaryTraces_() {

    assert(faceLabels.size() == baseElem.getGeometry().getNumFaces());

    for (size_t i = 0; i < faceLabels.size(); ++i) {
      const P1nDTrace<NF>* fTrace =
          makeTrace(baseElem, faceLabels[i], shapeType);
      addFace(fTrace, i);
    }

    normals.resize(getNumTraceFaces());

    for (size_t i = 0; i < getNumTraceFaces(); ++i) {
      baseElem.getGeometry().computeNormal(getTraceFaceIds()[i], normals[i]);
    }
  }

  P1nDBoundaryTraces(const P1nDBoundaryTraces<NF>& that)
      : ElementBoundaryTraces_(that), normals(that.normals) {}

  const VecDouble& getNormal(size_t FaceIndex) const {
    return normals[FaceIndex];
  }

protected:
  virtual const P1nDTrace<NF>* makeTrace(const P1nDElement<NF>& baseElem,
                                         const FaceLabel& flabel,
                                         const ShapeType& shType) const = 0;
};

/**
 \brief StandardP1nDMap class: standard local to global map for 2D/3D elements
 with linear shape functions

 StandardP1nDMap assumes that\n
 1) The GlobalNodalIndex of a node is an size_t\n
 2) All degrees of freedom are associated with nodes, and their values for each
 node ordered consecutively according to the field number.

 Consequently, the GlobalDofIndex of the degrees of freedom of node N with
 NField fields are given by

 (N-1)*NF + field-1, where \f$ 1 \le \f$ fields \f$ \le \f$ NF
 */

class StandardP1nDMap : public LocalToGlobalMap {
private:
  const std::vector<Element*>& elementArray;

public:
  StandardP1nDMap(const std::vector<Element*>& _elementArray)
      : LocalToGlobalMap(), elementArray(_elementArray) {}

  StandardP1nDMap(const StandardP1nDMap& that)
      : LocalToGlobalMap(that), elementArray(that.elementArray) {}

  virtual StandardP1nDMap* clone() const { return new StandardP1nDMap(*this); }

  inline GlobalDofIndex map(size_t field, size_t dof,
                            const GlobalElementIndex& ElementMapped) const {
    const Element* elem = elementArray[ElementMapped];
    // we subtract 1 from node ids in 1-based node numbering
    // return elem->getNumFields () * (elem-> getGeometry ().getConnectivity
    // ()[dof] - 1) + field; no need to subtract 1 with 0-based node numbering
    return elem->getNumFields() * (elem->getGeometry().getConnectivity()[dof]) +
           field;
  }

  inline size_t getNumElements() const { return elementArray.size(); }

  inline size_t getNumFields(const GlobalElementIndex& ElementMapped) const {
    return elementArray[ElementMapped]->getNumFields();
  }
  inline size_t getNumDof(const GlobalElementIndex& ElementMapped,
                          size_t field) const {
    return elementArray[ElementMapped]->getDof(field);
  }

  size_t getTotalNumDof() const {
    GlobalNodalIndex MaxNodeNumber = 0;

    for (size_t e = 0; e < elementArray.size(); e++) {
      const VecSize_t& conn = elementArray[e]->getGeometry().getConnectivity();

      for (size_t a = 0; a < conn.size(); ++a) {
        if (conn[a] > MaxNodeNumber) {
          MaxNodeNumber = conn[a];
        }
      }
    }

    // return maxNode * elementArray.get (0).numFields ();
    // add 1 here since nodes are number 0 .. numNodes-1 in 0-based node
    // numbering
    return static_cast<size_t>(MaxNodeNumber + 1) *
           elementArray[0]->getNumFields();
  }

protected:
  //! Access to ElementArray  for derived classes
  const std::vector<Element*>& getElementArray() const { return elementArray; }
};

#endif /* P1NDELEMENT_H_ */
