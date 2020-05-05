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
 * BasisFunctions.h
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

#ifndef BASISFUNCTIONS
#define BASISFUNCTIONS

#include "Shape.h"
#include "Quadrature.h"
#include "ElementGeometry.h"

/**
   \brief
   BasisFunctions: Evaluation of basis functions and derivatives at
   the quadrature points. Abstract class.

   A BasisFunctions object consists of:\n
   1) A set of quadrature points with quadrature weights\n
   2) A set of basis functions and their derivatives evaluated at these points\n


   \todo So far the number of spatial dimensions, needed to transverse
   the getDShapes and getQuadraturePointArrays is not provided, but should be
   obtained from the ElementGeometry where these basis functions are in. One
   way in which this may be computed is using the fact that
   getQuadraturePointCoordinates().size()/getIntegrationWeights().size() =
   getDShapes().size()/getShapes().size() = spatial dimensions

*/

class BasisFunctions {
public:
  inline BasisFunctions() {}
  inline virtual ~BasisFunctions() {}
  inline BasisFunctions(const BasisFunctions&) {}
  virtual BasisFunctions* clone() const = 0;

  //!  Shape functions at quadrature points
  //!  getShapes()[q*getBasisDimension()+a]
  //!  gives the value of shape function a at quadrature point q
  //!
  //!  getShapes returns an empty vector if no shape functions are available
  virtual const VecDouble& getShapes() const = 0;

  //! Derivatives of shape functions at quadrature points
  //! getDShapes()[q*getBasisDimension()*getNumberOfDerivativesPerFunction()+
  //! +a*getNumberOfDerivativesPerFunction()+i] gives the
  //! derivative in the i-th direction of degree of freedom a at quadrature
  //! point q
  //!
  //! getDShapes returns an empty vector if no derivatives are
  //! available
  virtual const VecDouble& getDShapes() const = 0;

  //! @return vector of integration weights
  virtual const VecDouble&
  getIntegrationWeights() const = 0; //!< Integration weights

  //! Coordinates of quadrature points in the real configuration
  //! getQuadraturePointCoordinates()
  //! q*ElementGeometry::getEmbeddingDimension()+i]
  //! returns the i-th coordinate in real space of quadrature point q
  virtual const VecDouble& getQuadraturePointCoordinates() const = 0;

  //! returns the number of shape functions provided
  virtual size_t getBasisDimension() const = 0;

  //! returns the number of directional derivative for each shape function
  virtual size_t getNumberOfDerivativesPerFunction() const = 0;

  //! returns the number of  number of coordinates for each Gauss point
  virtual size_t getSpatialDimensions() const = 0;
};

/**
   \brief dummy set with no basis functions.

   This class contains only static data and has the mission of providing
   a BasisFunctions object that has no basis functions in it.

   This becomes useful, for example, as a cheap way of providing Element
   with a BasisFunction object that can be returned but that occupies no memory.
   Since Element has to be able to have a BasisFunction object per field, by
   utilizing this object there is no need to construct an odd order for the
   fields in order to save memory.

 */
class EmptyBasisFunctions : public BasisFunctions {
public:
  inline EmptyBasisFunctions() {}
  inline virtual ~EmptyBasisFunctions() {}
  inline EmptyBasisFunctions(const EmptyBasisFunctions&) {}
  virtual EmptyBasisFunctions* clone() const {
    return new EmptyBasisFunctions(*this);
  }

  const VecDouble& getShapes() const { return ZeroSizeVector; }
  const VecDouble& getDShapes() const { return ZeroSizeVector; }
  const VecDouble& getIntegrationWeights() const { return ZeroSizeVector; }
  const VecDouble& getQuadraturePointCoordinates() const {
    return ZeroSizeVector;
  }
  size_t getBasisDimension() const { return 0; }
  size_t getNumberOfDerivativesPerFunction() const { return 0; }
  size_t getSpatialDimensions() const { return 0; }

private:
  static VecDouble ZeroSizeVector;
};

#endif
