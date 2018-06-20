/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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
 * Quadrature.h
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

#ifndef QUADRATURE
#define QUADRATURE
#include <algorithm>

/**
   \brief
   base class for any quadrature rule

   A quadrature rule provides and approximation of the integral over a domain
   \f$\Omega\f$, and it has:\n
          1) A number of quadrature point coordinates in \f$\Omega\f$. (map)
      2) Weights at each quadrature point
      3) A second set of coordinates for the same quadrature points. (Shape)

   The number of coordinates in (1) of each quadrature point should be
   unisolvent, i.e., each coordinate can be varied independently. The reason for
   this choice is that otherwise we need a way to convey the constraint between
   coordinates when functions of these coordinates and their derivatives are
   considered. For example, the mapping from the parametric configuration to the
   real space in finite elements.

   In addition to the unisolvent set of coordinates above, the class also allows
   for an alternative set of coordinates for each gauss point. These do not need
   to be unisolvent, and can be used to localize the quadrature points when
   embedded in a higher dimensional space. For example, if \f$\Omega\f$ is a
   triangle in 3D, then the second set of coordinates provide the coordinates of
   these points in 3D.

   A specific example is the following: consider a triangle in 2D, and
   the integration over a segment on its boundary. The quadrature
   object for a segment should have only one (map) coordinate in (1)
   to be unisolvent. This coordinate is one of the barycentric
   coordinates of the quadrature point over the segment. However, if
   we have a function over the triangle and would like to restrict its
   values to the segment, we need the coordinates of the quadrature
   points in the triangle. This is the case when integrating a shape
   function in the triangle over the segment. This second set of
   coordinates is given in (3).

   Based on this example we shall call the coordinates in (1)  map coordinates,
   and the coordinates in (3)  Shape coordinates.

   Although the coordinates in (3) are not  a natural inclusion in a quadrature
   rule, they provide the simplest implementation for the scenario just
   described.

   A number of specific Quadrature objects are defined in
   Quadrature.cpp, and declared here.

*/

class Quadrature {
public:
  //! @param NQ  number of quadrature points
  //! @param NC  number of coordinates for each point
  //! @param xqdat vector with coordinates of the quadrature points
  //! xqdat[a*NC+i] gives the i-th coordinate of quadrature point a
  //! @param wqdat vector with quadrature point weights
  //! wq[a] is the weight of quadrature point a
  //! This constructor sets the two sets of coordinates for the quadrature
  //! points to be the same.
  Quadrature(const double* const xqdat, const double* const wqdat,
             const size_t NC, const size_t NQ);
  //! @param NQ  number of quadrature points
  //! @param NCmap  number of map coordinates for each point
  //! @param NCshape  number of shape coordinates for each point
  //! @param xqdatmap vector with coordinates of the quadrature points
  //! xqdatmap[a*NC+i] gives the i-th coordinate of quadrature point a
  //! @param xqdatshape vector with coordinates of the quadrature points
  //! xqdatshape[a*NC+i] gives the i-th coordinate of quadrature point a
  //! @param wqdat vector with quadrature point weights
  //! wq[a] is the weight of quadrature point a
  Quadrature(const double* const xqdatmap, const double* const xqdatshape,
             const double* const wqdat, const size_t NCmap,
             const size_t NCshape, const size_t NQ);
  inline virtual ~Quadrature() {
    if (xqmap != xqshape) {
      delete[] xqshape;
      xqshape = NULL;
    }
    delete[] xqmap;
    xqmap = NULL;
    delete[] wq;
    wq = NULL;
  }

  Quadrature(const Quadrature&);
  Quadrature* clone() const;

  // Accessors/Mutators
  inline size_t getNumQuadraturePoints() const { return numQuadraturePoints; }

  //! Returns the number of map coordinates
  inline size_t getNumCoordinates() const { return numMapCoordinates; }

  //! Returns the number of shape coordinates
  inline size_t getNumShapeCoordinates() const { return numShapeCoordinates; }

  //! Return map coordinates of quadrature point q
  inline const double* getQuadraturePoint(size_t q) const {
    return xqmap + q * numMapCoordinates;
  }

  //! Return shape coordinates of quadrature point q
  inline const double* getQuadraturePointShape(size_t q) const {
    return xqshape + q * numShapeCoordinates;
  }

  //! Returns weight of quadrature point q
  inline double getQuadratureWeights(size_t q) const { return wq[q]; }

private:
  double* xqmap;
  double* xqshape;
  double* wq;
  size_t numMapCoordinates;
  size_t numShapeCoordinates;
  size_t numQuadraturePoints;
};

/**
   \brief SpecificQuadratures: class used just to qualify all specific
   quadrature objects used to build the quadrature rules.
 */
class SpecificQuadratures {};

/**
   \brief 3-point Gauss quadrature coordinates in the triangle (0,0), (1,0),
   (0,1), and its traces. Barycentric coordinates used for the Gauss points.
 */

class Triangle_1 : public SpecificQuadratures {
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;

  //! Face (1,2) quadrature
  static const Quadrature* const FaceOne;
  //! Face (2,3) quadrature
  static const Quadrature* const FaceTwo;
  //! Face (3,1) quadrature
  static const Quadrature* const FaceThree;

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
};

/**
   \brief 2-point Gauss quadrature coordinates in the segment (0,1).
   Barycentric coordinates used for the Gauss points.
 */
class Line_1 : public SpecificQuadratures {
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;

private:
  static const double BulkCoordinates[];
  static const double BulkWeights[];
};

/*!
 * \brief Class for 4 point quadrature rules for tetrahedra.
 *
 * 4-point Gauss quadrature coordinates in the tetrahedron with
 * 0(1,0,0), 1(0,1,0), 2(0,0,0), 3(0,0,1) as vertices.
 * Barycentric coordinates are used for the Gauss points.
 * Barycentric coordinates are specified with respect to vertices 1,2 and 4
 * in that order. Coordinate of vertex 3 is not independent.
 *
 * Quadrature for Faces:
 * Faces are ordered as -
 * Face 1: 2-1-0,
 * Face 2: 2-0-3,
 * Face 3: 2-3-1,
 * Face 4: 0-1-3.
 *
 * \todo Need to include a test for this quadrature
 */

class Tet_1 : public SpecificQuadratures {
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

#endif
