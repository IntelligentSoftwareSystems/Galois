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
 * Segment.h: a line segment
 * DG++
 *
 * Created by Adrian Lew on 10/7/06.
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

#ifndef SEGMENT
#define SEGMENT

#include "AuxDefs.h"
#include "ElementGeometry.h"

#include <cmath>
#include <iostream>
#include <cassert>

/**
   \brief Segment: Geometry of straight segments

   A Segment is:\n
   1) A set of indices that describe the connectivity of the segment,
   properly oriented. The coordinates
   are not stored in the element but wherever the application decides\n
   2) An affine map from a one-dimensional segment (parametric configuration)
   with length 1 to the convex
   hull of the two vertices. Segments embedded in two- and three-dimensional
   space are hence easily handled. \n

   The parametric configuration is the segment (0,1).\n
   The parametric coordinate used is the distance to 0.

   \warning Neither map nor dMap check for bounds of
   their array arguments

*/

template <size_t SPD>
class Segment : public AbstractGeom<SPD> {
public:
  Segment(const VecDouble& globalCoordVec, const VecSize_t& connectivity)
      : AbstractGeom<SPD>(globalCoordVec, connectivity) {
    assert(connectivity.size() == 2);
  }

  inline virtual ~Segment() {}

  Segment(const Segment<SPD>& that) : AbstractGeom<SPD>(that) {}

  virtual Segment<SPD>* clone() const { return new Segment<SPD>(*this); }

  inline size_t getNumVertices() const { return 2; }

  inline const std::string getPolytopeName() const { return "SEGMENT"; }

  inline size_t getParametricDimension() const { return 1; }

  inline size_t getEmbeddingDimension() const { return SPD; }

  //! @param X first parametric coordinate
  //! @param Y Output the result of the map
  void map(const double* X, double* Y) const;
  //! @param X first parametric coordinate
  //! @param DY Output the derivative of the map
  //! @param Jac Output the jacobian of the map
  void dMap(const double* X, double* DY, double& Jac) const;
  inline size_t getNumFaces() const { return 2; }

  //! \warning not implemented
  ElementGeometry* getFaceGeometry(size_t e) const {
    std::cerr << "Segment<SPD>::getFaceGeometry. "
                 "Not implemented!\n\n";
    return 0;
  }

  double getInRadius(void) const {
    double l;
    l = 0.0;
    for (size_t i = 0; i < SPD; i++) {
      l += (AbstractGeom<SPD>::getCoordinate(1, i) -
            AbstractGeom<SPD>::getCoordinate(0, i)) *
           (AbstractGeom<SPD>::getCoordinate(1, i) -
            AbstractGeom<SPD>::getCoordinate(0, i));
    }

    return (0.5 * sqrt(l));
  }

  double getOutRadius(void) const { return (getInRadius()); }

  virtual void computeNormal(size_t e, VecDouble& vNormal) const {
    std::cerr << "Segment::computeNormal not implemented yet" << std::endl;
    abort();
  }
};

// Class implementation

template <size_t SPD>
void Segment<SPD>::map(const double* X, double* Y) const {
  for (size_t i = 0; i < SPD; i++)
    Y[i] = X[0] * AbstractGeom<SPD>::getCoordinate(0, i) +
           (1 - X[0]) * AbstractGeom<SPD>::getCoordinate(1, i);

  return;
}

template <size_t SPD>
void Segment<SPD>::dMap(const double* X, double* DY, double& Jac) const {
  for (size_t i = 0; i < SPD; i++)
    DY[i] = AbstractGeom<SPD>::getCoordinate(0, i) -
            AbstractGeom<SPD>::getCoordinate(1, i);

  double g11 = 0;

  for (size_t i = 0; i < SPD; i++)
    g11 += DY[i] * DY[i];

  Jac = sqrt(g11);

  return;
}

#endif
