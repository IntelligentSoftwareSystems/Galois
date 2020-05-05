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
 * QuadratureP3.h
 * DG++
 *
 * Created by Adrian Lew on 10/21/06.
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

#ifndef QUADRATUREP3
#define QUADRATUREP3

#include "Quadrature.h"

/*!
 * \brief Class for 4 point quadrature rules for tetrahedrons.
 *
 * 4-point Gauss quadrature coordinates in the tetrahedron with
 * 0(1,0,0), 1(0,1,0), 2(0,0,0), 3(0,0,1) as vertices.
 * Barycentric coordinates are used for the Gauss points.
 * Barycentric coordinates are specified with respect to vertices 1,2 and 4
 * in that order.Coordinate of vertex 3 is not independent.
 *
 * Quadrature for Faces:
 * Faces are ordered as -
 * Face 1: 2-1-0,
 * Face 2: 2-0-3,
 * Face 3: 2-3-1,
 * Face 4: 0-1-3.
 */

class Tet_4Point : public SpecificQuadratures {
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;

  //! Face (2-1-0) quadrature
  static const Quadrature* const FaceOne;
  //! Face (2-0-3) quadrature
  static const Quadrature* const FaceTwo;
  //! Face (2-3-1) quadrature
  static const Quadrature* const FaceThree;
  //! Face (0-1-3) quadrature
  static const Quadrature* const FaceFour;

private:
  static const double BulkCoordinates[];
  static const double BulkWeights[];
  static const double FaceMapCoordinates[];
  static const double FaceOneShapeCoordinates[];
  static const double FaceOneWeights[];
  static const double FaceTwoShapeCoordinates[];
  static const double FaceTwoWeights[];
  static const double FaceThreeShapeCoordinates[];
  static const double FaceThreeWeights[];
  static const double FaceFourShapeCoordinates[];
  static const double FaceFourWeights[];
};

//! \brief 11 point quadrature rule for tetrahedron
//! Degree of precision 4, number of points 11.
class Tet_11Point : public SpecificQuadratures {
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;

  //! \todo Include face quadrature rules if needed.

private:
  static const double BulkCoordinates[];
  static const double BulkWeights[];
};

//! \brief 15 point quadrature rule for tetrahedron
//! Degree of precision 5, number of points 15.
class Tet_15Point : public SpecificQuadratures {
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;

  //! \todo Include face quadrature rules if needed.

private:
  static const double BulkCoordinates[];
  static const double BulkWeights[];
};

#endif
// Sriramajayam
