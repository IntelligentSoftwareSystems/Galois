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
 * ElementBoundaryTraces.h
 * DG++
 *
 * Created by Adrian Lew on 10/12/06.
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

#ifndef ELEMENTBOUNDARYTRACES
#define ELEMENTBOUNDARYTRACES

#include <vector>
#include "AuxDefs.h"
#include "Element.h"

/**
    \brief ElementBoundaryTraces: values of the trace of some
    or all the fields in an Element over some or all faces of the
    polytope.

    An ElementBoundaryTraces object contains\n
    1) The outward normal to the faces for which values are desired.\n
    2) The trace at these faces of some or all of the fields in the
       element. These traces are provided as Element objects, one for
       each face.

    The number of faces or fields in each face depends on the
    particular ElementBoundaryTraces object build. The exact
    number of fields whose trace is computed for each face will be
    determined by the Element object in each face.

    ElementBoundaryTraces objects are designed to work jointly
    with Element objects, so for example, it has no access to the
    ElementGeometry. The outward normal to the faces are included here
    because these are often used in integrands over faces.

    Each polytope has a convention to label ALL of its faces. By convention,
    these labels are consecutive integers starting at 0 to the total number of
   faces in the polytope minus one.
*/

class ElementBoundaryTraces {
public:
  ElementBoundaryTraces() {}
  virtual ~ElementBoundaryTraces() {}
  ElementBoundaryTraces(const ElementBoundaryTraces&) {}
  virtual ElementBoundaryTraces* clone() const = 0;

  //! Number of faces for which traces are provided
  virtual size_t getNumTraceFaces() const = 0;

  //! Returns the face number in the polytope whose traces are provided.
  //! Each polytope has a convention to label ALL of its faces. Traces are
  //! provided for a subset of these faces. A total number of getNumTraceFaces()
  //! faces have their traces in this object. getTraceFaceIds()[FaceIndex],
  //! with FaceIndex between
  //! 0 and getNumTraceFaces()-1, provides the face number in the polytope
  //! face element accesses with getTrace(FaceIndex).
  //!
  //! The value returned starts at 0 for the first face and so on.
  virtual const VecSize_t& getTraceFaceIds() const = 0;

  //! Returns the Trace number where the trace for face FaceIndex is stored.
  //! If no trace is provided for that face returns a -1.
  //!
  //! It is always true that FaceNumberToTrace[ getTraceFaceIds()[i] ] = i;
  //! for 0<= i <= getNumTraceFaces()-1
  virtual size_t getTraceNumberOfFace(size_t FaceIndex) const = 0;

  //! Returns a constant reference to the Element that contains
  //! the traces of the face getTraceFacesNumbers()[FaceIndex]. \n
  //! FaceIndex ranges from 0
  //! to the getNumTraceFaces()-1.
  virtual const Element& getTrace(size_t FaceIndex) const = 0;

  //! Returns getTrace(FaceIndex). Done for simplicity of the interface.
  //! FaceIndex ranges from 0
  //! to the getNumTraceFaces()-1.
  inline const Element& operator[](size_t FaceIndex) {
    return getTrace(FaceIndex);
  }

  //! Returns the outward normal to face getTraceFaceIds(FaceIndex)
  //! FaceIndex ranges from 0
  //! to the getNumTraceFaces()-1.
  virtual const VecDouble& getNormal(size_t FaceIndex) const = 0;

  //! map between the degrees of freedom of field in a trace
  //! and those in the original element
  //!
  //! @param FaceIndex starting from 0
  //! @param field  field number to map, starting from 0
  //! @param dof degree of freedom number on the trace of field "field"
  //!
  //! The function returns the degree of freedom number in the original
  //! element
  virtual size_t dofMap(size_t FaceIndex, size_t field, size_t dof) const = 0;
};

/**
   \brief ElementBoundaryTraces_: implementation of
   ElementBoundaryTraces

   An ElementBoundaryTraces_ allows derived classes to add
   Element_ objects, one per face whose trace is desired.

   The class is abstract since the getNormal function is yet to be
   defined by the specific derived ElementBoundaryTraces
   classes.

   The faces added with addFace are copied into the object, not
   referenced.

*/
class ElementBoundaryTraces_ : public ElementBoundaryTraces {
private:
  void copy(const ElementBoundaryTraces_& that) {
    FaceNumbers = that.FaceNumbers;

    for (size_t i = 0; i < that.FaceElements.size(); i++) {
      FaceElements.push_back(that.FaceElements[i]->clone());
    }
  }

  void destroy() {
    for (size_t i = 0; i < FaceElements.size(); i++) {
      delete FaceElements[i];
      FaceElements[i] = NULL;
    }
  }

public:
  ElementBoundaryTraces_() {}

  virtual ~ElementBoundaryTraces_() { destroy(); }

  ElementBoundaryTraces_(const ElementBoundaryTraces_& that)
      : ElementBoundaryTraces(that) {

    copy(that);
  }

  ElementBoundaryTraces_& operator=(const ElementBoundaryTraces_& that) {
    if (this != &that) {
      destroy();
      copy(that);
    }
    return (*this);
  }

  virtual ElementBoundaryTraces_* clone() const = 0;

  size_t getNumTraceFaces() const { return FaceElements.size(); }

  const VecSize_t& getTraceFaceIds() const { return FaceNumbers; }

  inline size_t getTraceNumberOfFace(size_t FaceIndex) const {
    for (size_t i = 0; i < FaceNumbers.size(); i++) {
      if (FaceNumbers[i] == FaceIndex) {
        return i;
      }
    }
    return -1;
  }

  virtual const Element& getTrace(size_t FaceIndex) const {
    return *FaceElements[FaceIndex];
  }

protected:
  void addFace(const Element_* NewFace, const size_t FaceNumber) {
    FaceElements.push_back(NewFace->clone());
    FaceNumbers.push_back(FaceNumber);
  }

private:
  std::vector<const Element*> FaceElements;
  VecSize_t FaceNumbers;
};

#endif
