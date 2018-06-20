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
 * testAssemble.cpp
 * DG++
 *
 * Created by Adrian Lew on 10/27/06.
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

#include "Triangle.h"
#include "ElementalOperation.h"
#include "StressWork.h"
#include "P12DElement.h"
#include "petscvec.h"
#include "petscmat.h"

static char help[] = "test";

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, (char*)0, help);

  double Vertices[] = {1, 0, 0, 1, 0, 0, 1, 1, 0, 2};
  std::vector<double> Vertices0(Vertices, Vertices + 10);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices0);

  std::vector<Element*> LocalElements;
  std::vector<DResidue*> LocalOperations(3);

  double conn[] = {1, 2, 3, 1, 4, 2, 2, 4, 5};

  int dim = 10;

  sleep(1);

  NeoHookean NH(1, 1);

  for (int e = 0; e < 3; e++) {
    LocalElements.push_back(
        new P12D<2>::Bulk(conn[3 * e], conn[3 * e + 1], conn[3 * e + 2]));
    LocalOperations[e] = new StressWork(LocalElements[e], NH, 0, 1);
  }
  StandardP12DMap L2G(LocalElements);

  Vec Dofs, resVec;
  Mat dresMat;

  VecCreate(PETSC_COMM_WORLD, &Dofs);
  VecSetSizes(Dofs, PETSC_DECIDE, dim);
  VecSetFromOptions(Dofs);
  VecDuplicate(Dofs, &resVec);

  MatCreateSeqDense(PETSC_COMM_SELF, dim, dim, PETSC_NULL, &dresMat);
  MatSetOption(dresMat, MAT_SYMMETRIC);

  double dofvalues[] = {0.1, 0., 0., 0.1, 0., 0., 0.1, 0.2, -0.1, 0.2};
  int indices[]      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  VecSetValues(Dofs, dim, indices, dofvalues, INSERT_VALUES);
  VecAssemblyBegin(Dofs);
  VecAssemblyEnd(Dofs);

  DResidue::assemble(LocalOperations, L2G, Dofs, &resVec, &dresMat);

  VecAssemblyBegin(resVec);
  VecAssemblyEnd(resVec);
  MatAssemblyBegin(dresMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(dresMat, MAT_FINAL_ASSEMBLY);

  VecView(resVec, PETSC_VIEWER_STDOUT_SELF);
  MatView(dresMat, PETSC_VIEWER_STDOUT_SELF);

  std::cout << "\n\nConsistency Test\n";

  Vec resVecPlus;
  Vec resVecMinus;
  VecDuplicate(Dofs, &resVecPlus);
  VecDuplicate(Dofs, &resVecMinus);

  Mat dresMatNum;
  MatDuplicate(dresMat, MAT_DO_NOT_COPY_VALUES, &dresMatNum);

  double EPS = 1.e-6;
  for (int i = 0; i < dim; i++) {
    double v[1];
    int ix[1] = {i};

    VecGetValues(Dofs, 1, ix, v);

    double ival   = v[0];
    double newval = ival + EPS;

    VecSetValue(Dofs, i, newval, INSERT_VALUES);
    VecAssemblyBegin(Dofs);
    VecAssemblyEnd(Dofs);

    DResidue::assemble(LocalOperations, L2G, Dofs, &resVecPlus, 0);

    newval = ival - EPS;

    VecSetValue(Dofs, i, newval, INSERT_VALUES);
    VecAssemblyBegin(Dofs);
    VecAssemblyEnd(Dofs);

    DResidue::assemble(LocalOperations, L2G, Dofs, &resVecMinus, 0);

    newval = ival;

    VecSetValue(Dofs, i, newval, INSERT_VALUES);
    VecAssemblyBegin(Dofs);
    VecAssemblyEnd(Dofs);

    VecAssemblyBegin(resVecPlus);
    VecAssemblyEnd(resVecPlus);
    VecAssemblyBegin(resVecMinus);
    VecAssemblyEnd(resVecMinus);

    double *Plus, *Minus;
    VecGetArray(resVecPlus, &Plus);
    VecGetArray(resVecMinus, &Minus);

    for (int j = 0; j < dim; j++) {
      double deriv = (Plus[j] - Minus[j]) / (2 * EPS);
      MatSetValue(dresMatNum, i, j, deriv, INSERT_VALUES);
    }
    VecRestoreArray(resVecPlus, &Plus);
    VecRestoreArray(resVecMinus, &Minus);
  }

  MatAssemblyBegin(dresMatNum, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(dresMatNum, MAT_FINAL_ASSEMBLY);

  // Compare matrices
  double error = 0;
  double norm  = 0;

  double* dresMatA;
  MatGetArray(dresMat, &dresMatA);
  double* dresMatNumA;
  MatGetArray(dresMatNum, &dresMatNumA);

  for (int i = 0; i < dim * dim; i++) {
    error += pow(dresMatA[i] - dresMatNumA[i], 2);
    norm += pow(dresMatA[i], 2);
  }

  MatRestoreArray(dresMat, &dresMatA);
  MatRestoreArray(dresMatNum, &dresMatNumA);

  error = sqrt(error);
  norm  = sqrt(norm);

  if (error / norm < EPS * 100)
    std::cout << "Consistency test successful - norm = " << norm
              << " error = " << error << "\n";
  else
    std::cout << "Consistency test failed - norm = " << norm
              << " error = " << error << "\n";

  VecDestroy(resVecPlus);
  VecDestroy(resVecMinus);
  MatDestroy(dresMatNum);

  VecDestroy(Dofs);
  VecDestroy(resVec);
  MatDestroy(dresMat);
}
