/**
 * P12DElement.h: 2D Element with linear shape functions
 *
 * DG++
 *
 * Created by Adrian Lew on 9/22/06.
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

#ifndef P12DELEMENT
#define P12DELEMENT

#include "Element.h"
#include "P1nDElement.h"

#include "ElementGeometry.h"
#include "Triangle.h"
#include "ElementBoundaryTrace.h"
#include "ShapesEvaluated.h"

#include <iostream>
#include <vector>
#include <cassert>

/**
 \brief Two-dimensional linear triangles with NF different fields
 */

template<size_t NF>
class P12DElement: public P1nDElement<NF> {
public:
  P12DElement (const Triangle<2>& _elemGeom) :
    P1nDElement<NF> (_elemGeom) {
    ShapesP12D::Bulk modelShape (_elemGeom);
    Element_::addBasisFunctions (modelShape);
  }

  P12DElement (const P12DElement<NF> &that) :
    P1nDElement<NF> (that) {

  }

  virtual P12DElement<NF>* clone () const {
    return new P12DElement<NF> (*this);
  }
};

/**
 \brief P12DTrace: traces of P12DElements

 These trace elements are templated according to the number of Linear
 fields and the Triangle face P12DTrace<NF>::FaceLabel on which the
 trace is computed.

 Traces of P12D elements are somewhat special, since for each side one of the
 bulk basis functions is identically zero, effectively leaving only two degrees
 of freedom per face. To reduce the amount of memory allocated per element is then
 convenient to consider Trace elements with only two degrees of freedom per field
 per face, i.e., use the ShapesP12D::Faces ShapesEvaluated objects.

 However, ocassionally it may be necessary to have the boundary elements have the same
 number of degrees of freedom as the bulk elements, as it is generally the case for
 arbitrary bases. Hence, during the construction of these elements it is possible
 to decide which type of shape functions to use. This is accomplished by specifying
 the type ShapeType.

 */

template<size_t NF>
class P12DTrace: public P1nDTrace<NF> {
public:

  //! @param BaseElement Element whose trace is desired
  //! @param FaceName Face on which to take the trace
  //! @param Type ShapeType of face, i.e., with two or three dof
  P12DTrace (const P12DElement<NF> & BaseElement, const typename P1nDTrace<NF>::FaceLabel& FaceName,
      const typename P1nDTrace<NF>::ShapeType& Type);

  virtual ~P12DTrace () {
  }

  P12DTrace (const P12DTrace<NF> &that) :
    P1nDTrace<NF> (that) {
  }

  virtual P12DTrace<NF> * clone () const {
    return new P12DTrace<NF> (*this);
  }
private:
  //! check if the faceLabel and shapeType are consistent with
  //! 2D element trace
  //! @param faceLabel
  //! @param shapeType
  void checkArgs (const typename P1nDTrace<NF>::FaceLabel& faceLabel,
      const typename P1nDTrace<NF>::ShapeType& shapeType) {
    // Triangle has only 3 faces
    assert (faceLabel != P1nDTrace<NF>::FaceFour);

    // valid ShapeTypes are TwoDofs and ThreeDofs
    assert (shapeType == P1nDTrace<NF>::TwoDofs || shapeType == P1nDTrace<NF>::ThreeDofs);
  }

};

template<size_t NF>
P12DTrace<NF>::P12DTrace (const P12DElement<NF> & BaseElement,
    const typename P1nDTrace<NF>::FaceLabel& FaceName, const typename P1nDTrace<NF>::ShapeType& Type) :
  P1nDTrace<NF> (BaseElement) {

  checkArgs (FaceName, Type);

  const ElementGeometry& TriGeom = Element::getGeometry ();
  assert (dynamic_cast<const Triangle<2>* > (&TriGeom) != NULL);

  ElementGeometry* faceGeom = TriGeom.getFaceGeometry(FaceName);
  assert (dynamic_cast<Segment<2>* > (faceGeom) != NULL);

  if (Type == P1nDTrace<NF>::TwoDofs) {
    ShapesP12D::Faces modelShape (*faceGeom);
    Element_::addBasisFunctions (modelShape);

  } else {
    // Type==ThreeDofs
    switch (FaceName) {
      case P1nDTrace<NF>::FaceOne: {
        ShapesP12D::FaceOne ModelShape (*faceGeom);
        Element_::addBasisFunctions (ModelShape);
        break;
      }

      case P1nDTrace<NF>::FaceTwo: {
        ShapesP12D::FaceTwo ModelShape (*faceGeom);
        Element_::addBasisFunctions (ModelShape);
        break;
      }

      case P1nDTrace<NF>::FaceThree: {
        ShapesP12D::FaceThree ModelShape (*faceGeom);
        Element_::addBasisFunctions (ModelShape);
        break;
      }
    }
  }

  delete faceGeom; faceGeom = NULL;
}

/**
 \brief P12DElementBoundaryTraces:  group of traces of P12DElements.

 It contains P12DTrace Elements. It is possible to specify which faces to build  traces for.

 getTrace(i) returns the i-th face for which traces were built. The order of these faces is always increasing
 in face number. Example: if only faces one and three have traces, then getTrace(0) returns face one's trace,
 and getTrace(1) face three's trace.

 It does not make a copy or keep a reference of the BaseElement.

 */

template<size_t NF>
class P12DElementBoundaryTraces: public P1nDBoundaryTraces<NF> {
public:
  //! @param BaseElement Element for which traces are to be build
  //! @param flabels   a vector of face labels (size is 3 for triangles)
  //! @param shType type of trace element to use. See P12DTrace<NF>
  P12DElementBoundaryTraces (const P12DElement<NF> &BaseElement,
      const std::vector<typename P1nDTrace<NF>::FaceLabel>& flabels,
      const typename P1nDTrace<NF>::ShapeType& shType):
        P1nDBoundaryTraces<NF> (BaseElement, flabels, shType) {

  }

  virtual ~P12DElementBoundaryTraces () {
  }

  P12DElementBoundaryTraces (const P12DElementBoundaryTraces<NF> & OldElem) :
    P1nDBoundaryTraces<NF> (OldElem) {
  }

  P12DElementBoundaryTraces<NF> * clone () const {
    return new P12DElementBoundaryTraces<NF> (*this);
  }


  size_t dofMap (size_t FaceIndex, size_t field, size_t dof) const;

};

// Class Implementation
template<size_t NF> size_t P12DElementBoundaryTraces<NF>::dofMap (
    size_t FaceIndex, size_t field, size_t dof) const {
  size_t val;
  if (ElementBoundaryTraces::getTrace (FaceIndex).getDof (field) == 3) {
    val = dof;
  } else {
    // getTrace(FaceIndex).getDof(field)=2
    switch (ElementBoundaryTraces::getTraceFaceIds ()[FaceIndex]) {
      case 0:
        val = dof;
        break;

      case 1:
        val = dof + 1;
        break;

      case 2:
        val = (dof == 0 ? 2 : 0);
        break;

      default:
        std::cerr << "P12DElementBoundaryTraces.DofMap Error\n";
        exit (1);
    }
  }

  return val;
}

/**
 \brief P12D<NF> family of elements over triangles with NF linearly
 interpolated fields.
 */

template<size_t NF> class P12D: public SpecificElementFamily {
public:
  //! Linear element over a triangle
  typedef P12DElement<NF> Bulk;

  //! Linear elements over segments 
  typedef P12DTrace<NF> Face;

  //! Traces on the boundary for P12DElement<NF> 
  typedef P12DElementBoundaryTraces<NF> Traces;
};



#endif

