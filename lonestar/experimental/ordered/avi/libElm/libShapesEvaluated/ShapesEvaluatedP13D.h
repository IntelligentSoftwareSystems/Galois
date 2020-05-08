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
 * ShapesEvaluatedP13D.h
 * DG++
 *
 * Created by Ramsharan Rangarajan.
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

#ifndef SHAPESEVALUATEDP13D
#define SHAPESEVALUATEDP13D

#include "ShapesEvaluated.h"
#include "Quadrature.h"
#include "Linear.h"

/**
   \brief Shape functions for P13D elements: Linear functions on tetrahedra.

   It containes two types of traces
   1) ShapesP13D::Faces are linear functions on triangles and their dofs are
   those associated with the nodes of the triangular face.

   2) ShapesP13D::FaceOne, ShapesP13D::FaceTwo, ShapesP13D::FaceThree and
   ShapesP13D::FaceFour are the full set of linear shape functions in the
   elements evaluated at quadrature points in each one of the faces. Instead of
   having 3 dofs per field per face, there are 3 dofs per field per face.  Some
   of these values are trivially zero, i.e., those of the shape function
   associated to the node opposite to the face where the quadrature point is. As
   was the done in ShapesP12D, these are provided since ocassionally it may be
   necessary to have the same number of dofs as the bulk fields. In the most
   general case of arbitrary bases, there is generally no shape function that is
   zero on the face, and hence bulk and trace fields have the same number of
   dofs.
*/

class ShapesP13D : public SpecificShapesEvaluated {
protected:
  static const size_t bctMap[];

public:
  static const Shape* const P13D;
  static const Shape* const P12D;

  typedef ShapesEvaluated__<P13D, Tet_1::Bulk> Bulk;

  typedef ShapesEvaluated__<P12D, Triangle_1::Bulk> Faces;

  typedef ShapesEvaluated__<P13D, Tet_1::FaceOne> FaceOne;     // 2-1-0
  typedef ShapesEvaluated__<P13D, Tet_1::FaceTwo> FaceTwo;     // 2-0-3
  typedef ShapesEvaluated__<P13D, Tet_1::FaceThree> FaceThree; // 2-3-1
  typedef ShapesEvaluated__<P13D, Tet_1::FaceFour> FaceFour;   // 0-1-3
};

#endif
