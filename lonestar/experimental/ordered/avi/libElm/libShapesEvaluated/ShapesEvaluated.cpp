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

/*
 * ShapesEvaluatedImpl.cpp
 * DG++
 *
 * Created by Adrian Lew on 9/7/06.
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

#include "Quadrature.h"
#include "Linear.h"
#include "ShapesEvaluated.h"

#include "util.h"

// #include "petscblaslapack.h"

#include <iostream>

extern "C" void dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);

const Shape* const ShapesP12D::P12D = new Linear<2>;
const Shape* const ShapesP12D::P11D = new Linear<1>;

const Shape* const ShapesP11D::P11D = new Linear<1>;

void ShapesEvaluated::createObject(const ElementGeometry& EG) {
  const Shape& TShape           = accessShape();
  const Quadrature& TQuadrature = accessQuadrature();

  int Na = TShape.getNumFunctions();
  int Nq = TQuadrature.getNumQuadraturePoints();
  int Nd = EG.getEmbeddingDimension();
  int Np = EG.getParametricDimension();

  LocalShapes.resize(Nq * Na);
  if (Np == Nd)
    LocalDShapes.resize(Nq * Na * Nd);
  LocalWeights.resize(Nq);
  LocalCoordinates.resize(Nq * Nd);

  double* DY = new double[Nd * Np];
  double* Y  = new double[Nd];

  for (int q = 0; q < Nq; q++) {
    EG.map(TQuadrature.getQuadraturePoint(q), Y);
    for (int i = 0; i < Nd; i++)
      LocalCoordinates[q * Nd + i] = Y[i];

    for (int a = 0; a < Na; a++)
      LocalShapes[q * Na + a] =
          TShape.getVal(a, TQuadrature.getQuadraturePointShape(q));

    double Jac;
    EG.dMap(TQuadrature.getQuadraturePoint(q), DY, Jac);

    LocalWeights[q] = (TQuadrature.getQuadratureWeights(q)) * Jac;

    // Compute derivatives of shape functions only when the element map goes
    // between spaces of the same dimension
    if (Np == Nd) {
      double* DYInv = new double[Np * Np];
      double* DYT   = new double[Np * Np];

      // Lapack
      {
        // Transpose DY to Fortran mode
        for (int k = 0; k < Nd; k++)
          for (int M = 0; M < Np; M++) {
            DYT[k * Nd + M] = DY[M * Np + k];

            // Right hand-side
            DYInv[k * Nd + M] = k == M ? 1 : 0;
          }

        int* IPIV = new int[Np];
        int INFO;

        // DYInv contains the transpose of the inverse
        dgesv_(&Np, &Np, DYT, &Np, IPIV, DYInv, &Np, &INFO);

#ifdef DEBUG_TEST
        const double* ptCoord    = TQuadrature.getQuadraturePoint(q);
        const double* shapeCoord = TQuadrature.getQuadraturePointShape(q);

        printIter(std::cerr << "ptCoord = ", ptCoord, ptCoord + Np);
        printIter(std::cerr << "shapeCoord ", shapeCoord, shapeCoord + Np);
        printIter(std::cerr << "Y = ", Y, Y + Np);
        printIter(std::cerr << "DY = ", DY, DY + Np * Np);
        printIter(std::cerr << "DYInv = ", DYInv, DYInv + Np * Np);
        std::cerr << "----------------------" << std::endl;
#endif

        if (INFO != 0) {
          std::cerr << "ShapesEvaluated::CreateObject: Lapack could not invert "
                       "matrix\n";
          abort();
        }

#ifdef DEBUG_TEST // Output only useful during testing
        for (int r = 0; r < Nd * Np; r++)
          std::cout << DYInv[r] << " ";
        std::cout << "\n";
#endif

        delete[] DYT;
        delete[] IPIV;
      }

      for (int a = 0; a < Na; a++)
        for (int i = 0; i < Nd; i++) {
          LocalDShapes[q * Na * Nd + a * Nd + i] = 0;
          for (int M = 0; M < Np; M++)
            LocalDShapes[q * Na * Nd + a * Nd + i] +=
                TShape.getDVal(a, TQuadrature.getQuadraturePointShape(q), M) *
                DYInv[M * Np + i];
        }

      delete[] DYInv;
    } else {
      std::cerr << "did not enter lapack block" << std::endl;
    }
  }

#ifdef DEBUG_TEST
  printIter(std::cerr << "LocalDShapes = ", LocalDShapes.begin(),
            LocalDShapes.end());
#endif

  delete[] DY;
  delete[] Y;
}

ShapesEvaluated::ShapesEvaluated(const ShapesEvaluated& SE)
    : BasisFunctions(SE), LocalShapes(SE.LocalShapes),
      LocalDShapes(SE.LocalDShapes), LocalWeights(SE.LocalWeights),
      LocalCoordinates(SE.LocalCoordinates) {}
