/**
 * P13DElement.h: A 3D element with linear shape functions
 *
 * DG++
 *
 * Created by Ramsharan Rangarajan
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

#ifndef P13DELEMENT
#define P13DELEMENT

#include "P1nDElement.h"
#include "P12DElement.h"
#include "ShapesEvaluatedP13D.h"
#include "Tetrahedron.h"
#include "ElementGeometry.h"


#include <iostream>
#include <vector>
#include <cassert>

//! \brief three-dimensional linear tetrahedra with NF different fields.
template<size_t NF>
class P13DElement: public P1nDElement<NF> {
public:
  //! \param _elemGeom A reference to element geometry
  P13DElement (const Tetrahedron& _elemGeom): P1nDElement<NF> (_elemGeom) {
    ShapesP13D::Bulk modelShape (_elemGeom);
    Element_::addBasisFunctions (modelShape);
  }


  //! Copy constructor
  P13DElement (const P13DElement<NF>& that): P1nDElement<NF> (that) {

  }

  //! Cloning mechanism
  virtual P13DElement<NF>* clone () const {
    return new P13DElement<NF> (*this);
  }

};


// Class P13DTrace:
/** \brief Traces of P13DElement.

 These trace elements are templated according to the number of
 linear fields and the Tetrahedron face
 P13DTrace<NF>::FaceLabel on which the trace is computed.

 Traces of P13DElements are somewhat special, since for each side
 one of the bulk basis functions is identically zero, effectively
 leaving three dofs per face. To reduce the amount of memory
 allocated per element is then convenient to consider trace elements
 with only 3 dofs per field per face, i.e., use the
 ShapesP13D::Faces ShapesEvaluated objects.

 However, ocassionally, it may be necessary to have the boundary
 elements have the same nummber of dofs as the bulk elements, as it
 is generally the case for arbitrary bases. Hence during their
 construction, it is possible to specify what type of shape
 functions to use. This is accomplished by specifying the type
 ShapeType.

 \warning As opposed to P13DElement(s), there is no local copy of
 the geometry of the element here, but only a reference to
 it. Hence, the destruction of the base geometry before the
 destruction of the element will render the implementation
 inconsistent.

 \todo As was commented in P12DTrace, TetGeom may have to be a
 pointer and not a reference, since this is a reference to a copy of
 the original geometry.
 */
template<size_t NF>
class P13DTrace: public P1nDTrace<NF> {
public:

  //! \param BaseElement Element whose trace is desired.
  //! \param FaceName Face on which to take the trace.
  //! \param Type ShapeType of the face, i.e., with three or four dofs.
  P13DTrace (const P13DElement<NF> &BaseElement, const typename P1nDTrace<NF>::FaceLabel& FaceName,
      const typename P1nDTrace<NF>::ShapeType& Type);

  virtual ~P13DTrace () {
  }

  //! Copy constructor
  P13DTrace (const P13DTrace<NF> &OldElement_) :
    P1nDTrace<NF> (OldElement_) {
  }

  //! Cloning mechanism
  virtual P13DTrace<NF>* clone () const {
    return new P13DTrace<NF> (*this);
  }

private:
  //! checks if arguments are consistend with 3D element
  //! @param flabel 
  //! @param shType
  void checkArgs (const typename P1nDTrace<NF>::FaceLabel& flabel, const typename P1nDTrace<NF>::ShapeType& shType) {
    assert (shType != P1nDTrace<NF>::TwoDofs);
  }

};

// Implementation of class P13DTrace:

template<size_t NF>
P13DTrace<NF>::P13DTrace (const P13DElement<NF> &BaseElement,
    const typename P1nDTrace<NF>::FaceLabel& FaceName, const typename P1nDTrace<NF>::ShapeType& Type) :
  P1nDTrace<NF> (BaseElement) {

  checkArgs (FaceName, Type);

  const ElementGeometry& tetGeom = Element::getGeometry ();
  assert (dynamic_cast<const Tetrahedron*> (&tetGeom) != NULL);

  ElementGeometry* faceGeom = tetGeom.getFaceGeometry(FaceName);
  assert (dynamic_cast<Triangle<3>* > (faceGeom) != NULL);


  if (Type == P1nDTrace<NF>::ThreeDofs) {
    ShapesP13D::Faces ModelShape (*faceGeom);
    Element_::addBasisFunctions (ModelShape);
  } else {
    //Type == FourDofs
    switch (FaceName) {
      case P1nDTrace<NF>::FaceOne: {
        ShapesP13D::FaceOne ModelShape(*faceGeom);
        Element_::addBasisFunctions(ModelShape);
        break;
      }

      case P1nDTrace<NF>::FaceTwo: {
        ShapesP13D::FaceTwo ModelShape(*faceGeom);
        Element_::addBasisFunctions(ModelShape);
        break;
      }

      case P1nDTrace<NF>::FaceThree: {
        ShapesP13D::FaceThree ModelShape(*faceGeom);
        Element_::addBasisFunctions(ModelShape);
        break;
      }

      case P1nDTrace<NF>::FaceFour: {
        ShapesP13D::FaceFour ModelShape(*faceGeom);
        Element_::addBasisFunctions(ModelShape);
        break;
      }

    }
  }

  delete faceGeom; faceGeom = NULL;
}

// Class for ElementBoundaryTraces:

/**
 \brief Group of traces for P13DElement.

 It contains P13DTrace elements. It is possible to specify which faces to build traces for.

 getTrace(i) returns the i-th face for which traces are built. The order of these faces is always increasing
 in number.
 It does not make a copy o keep reference of the BaseElement.
 */

template<size_t NF>
class P13DElementBoundaryTraces: public P1nDBoundaryTraces<NF> {
public:

  //! \param BaseElement Element for which to build traces.
  //! \param faceLabels is a vector telling which faces to build
  //! \param Type type of trace element to use. 
  P13DElementBoundaryTraces (const P13DElement<NF> &BaseElement,
      const std::vector<typename P1nDTrace<NF>::FaceLabel>& faceLabels,
      typename P13DTrace<NF>::ShapeType Type):
        P1nDBoundaryTraces<NF> (BaseElement, faceLabels, Type) {

  }

  virtual ~P13DElementBoundaryTraces () {
  }

  //! Copy constructor
  P13DElementBoundaryTraces (const P13DElementBoundaryTraces<NF> &OldElem) :
    P1nDBoundaryTraces<NF> (OldElem) {
  }

  //! Cloning mechanism
  P13DElementBoundaryTraces<NF> * clone () const {
    return new P13DElementBoundaryTraces<NF> (*this);
  }

  //! map dofs between dofs of field in a trace and those in the original element.
  //! \param FaceIndex starting from 0.
  //! \param field field number to map, starting from 0.
  //! \param dof degree of freedom number on the trace of field "field"
  //! The function returns the degree of freedom number in the original element.
  size_t dofMap (size_t FaceIndex, size_t field, size_t dof) const;
};

// Implementation of class P13DElementBoundaryTraces
template<size_t NF>
size_t P13DElementBoundaryTraces<NF>::dofMap (size_t FaceIndex, size_t field, size_t dof) const {
  size_t val;

  if (ElementBoundaryTraces::getTrace (FaceIndex).getDof (field) == 4) {
    val = dof;
  } else { // Three dofs per face.
  
    const size_t* FaceNodes = Tetrahedron::FaceNodes;
    size_t facenum = ElementBoundaryTraces::getTraceFaceIds ()[FaceIndex];
    val = FaceNodes[3 * facenum + dof];
  }
  return val;
}

//! \brief Family of elements over tetrahedra with NF linearly interpolated fields.
template<size_t NF>
class P13D: public SpecificElementFamily {

public:
  //! Linear over the element.
  typedef P13DElement<NF> Bulk;

  //! Linear over triangles.
  typedef P13DTrace<NF> Face;

  //! Traces on the boundary of P13DElement<NF>
  typedef P13DElementBoundaryTraces<NF> Traces;

};



#endif

