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
 * Shape.h
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

#ifndef SHAPE
#define SHAPE

#include <cstddef>

/**
 \brief base class for any set of basis (or shape) functions and its
 first derivatives

 A set of basis functions permits the evaluation of any of the functions
 in the basis at any point

 Notice that two Shape classes differ if they  span the same space but
 have different bases.
 */

class Shape {
public:
  inline Shape() {}

  inline virtual ~Shape() {}

  inline Shape(const Shape&) {}

  virtual Shape* clone() const = 0;

  // Accessors/Mutators
  virtual size_t getNumFunctions() const = 0;
  virtual size_t
  getNumVariables() const = 0; //!< Number of arguments of the functions

  //! Value of shape \f$N_a(x)\f$
  //! @param a node id
  //! @param x coordinates of the point x
  //!
  //! We have purposedly left the type of coordinates the point should have
  //! unspecified, for flexibility. Barycentric and Cartesian coordinates are
  //! adopted thoughout the code.
  //!
  //! \todo It'd be nice to have some form of Coordinate object, which may
  //! derive Barycentric and Cartesian coordinates, and that would guarantee
  //! that the argument to each function is always the correct one.

  virtual double getVal(size_t a, const double* x) const = 0;

  //! Value of \f$\frac{\partial N_a}{\partial x_i}(x)\f$
  //! @param a node id
  //! @param x coordinates of the point a
  //! @param i coordinate number
  //!
  //! We have purposedly left the type of coordinates the point should have
  //! unspecified, for flexibility. Barycentric and Cartesian coordinates are
  //! adopted thoughout the code.
  virtual double getDVal(size_t a, const double* x, size_t i) const = 0;

  //! Consistency test for getVal and getDVal
  //! @param x coordinates of the point at which to test
  //! @param Pert size of the perturbation with which to compute numerical
  //! derivatives (x->x+Pert)
  bool consistencyTest(const double* x, const double Pert) const;
};

#endif
