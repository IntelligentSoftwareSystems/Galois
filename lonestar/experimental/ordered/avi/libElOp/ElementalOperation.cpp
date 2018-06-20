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
 * ElementalOperation.cpp
 * DG++
 *
 * Created by Adrian Lew on 10/25/06.
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

#include <cmath>
#include <vector>
#include <algorithm>

#include "util.h"
#include "ElementalOperation.h"

bool DResidue::consistencyTest(const DResidue& DRes,
                               const VecSize_t& DofPerField,
                               const MatDouble& argval) {
  size_t NFields = DRes.getFields().size();

  MatDouble largval(argval);
  MatDouble funcval;
  MatDouble funcvalplus;
  MatDouble funcvalminus;

  FourDVecDouble dfuncval;
  FourDVecDouble dfuncvalnum;

  const double EPS = 1.e-6;

  dfuncvalnum.resize(NFields);
  for (size_t f = 0; f < NFields; f++) {
    dfuncvalnum[f].resize(DofPerField[f]);
    for (size_t a = 0; a < DofPerField[f]; a++) {
      dfuncvalnum[f][a].resize(NFields);
      for (size_t g = 0; g < NFields; g++) {
        dfuncvalnum[f][a][g].resize(DofPerField[g]);
      }
    }
  }

  double maxval = 0;
  for (size_t f = 0; f < NFields; f++) {
    for (size_t a = 0; a < DofPerField[f]; a++) {
      if (maxval > fabs(argval[f][a])) {
        maxval = fabs(argval[f][a]);
      }
    }
  }

  maxval += 1;

  for (size_t f = 0; f < NFields; f++) {
    for (size_t a = 0; a < DofPerField[f]; a++) {
      double ival = largval[f][a];

      largval[f][a] = ival + EPS * maxval;
      DRes.getVal(largval, funcvalplus);

      largval[f][a] = ival - EPS * maxval;
      DRes.getVal(largval, funcvalminus);

      largval[f][a] = ival;

      for (size_t g = 0; g < NFields; g++) {
        for (size_t b = 0; b < DofPerField[g]; b++) {
          dfuncvalnum[g][b][f][a] =
              (funcvalplus[g][b] - funcvalminus[g][b]) / (2 * EPS * maxval);
        }
      }
    }
  }

  DRes.getDVal(largval, funcval, dfuncval);

  double error = 0;
  double norm  = 0;
  for (size_t f = 0; f < dfuncval.size(); f++) {
    for (size_t a = 0; a < dfuncval[f].size(); a++) {
      for (size_t g = 0; g < dfuncval[f][a].size(); g++) {
        for (size_t b = 0; b < dfuncval[f][a][g].size(); b++) {
          error += pow(dfuncval[f][a][g][b] - dfuncvalnum[f][a][g][b], 2);
          norm += pow(dfuncval[f][a][g][b], 2);
        }
      }
    }
  }

  error = sqrt(error);
  norm  = sqrt(norm);

  if (error / norm > EPS * 100) {
    std::cerr << "DResidue::ConsistencyTest. DResidue not consistent\n";
    std::cerr << "norm: " << norm << " error: " << error << "\n";
    return false;
  }
  return true;
}

enum AssembleMode {
  RESIDUE,
  DRESIDUE,
};

template <typename T>
static bool _Assemble(std::vector<T*>& DResArray, const LocalToGlobalMap& L2G,
                      const VecDouble& Dofs, VecDouble& ResVec,
                      MatDouble& DResMat, const AssembleMode& mode) {

  // VecZeroEntries(*ResVec);
  std::fill(ResVec.begin(), ResVec.end(), 0.0);

  if (mode == DRESIDUE) {
    // MatZeroEntries(DResMat);
    for (MatDouble::iterator i = DResMat.begin(); i != DResMat.end(); ++i) {
      std::fill(i->begin(), i->end(), 0.0);
    }
  }

  MatDouble argval;
  MatDouble funcval;
  FourDVecDouble dfuncval;

  // double * GDofs;
  // VecGetArray(Dofs, &GDofs);
  const VecDouble& GDofs = Dofs;

  for (size_t e = 0; e < DResArray.size(); e++) {
    const VecSize_t& DPF = DResArray[e]->getFields();
    size_t localsize     = 0;

    if (argval.size() < DPF.size()) {
      argval.resize(DPF.size());
    }

    for (size_t f = 0; f < DPF.size(); f++) {
      if (argval[f].size() < DResArray[e]->getFieldDof(f)) {
        argval[f].resize(DResArray[e]->getFieldDof(f));
      }

      localsize += DResArray[e]->getFieldDof(f);

      for (size_t a = 0; a < DResArray[e]->getFieldDof(f); a++) {
        argval[f][a] = GDofs[L2G.map(f, a, e)];
      }
    }

    if (mode == DRESIDUE) {
      // I am using a dynamic_cast to prevent writing two versions of
      // essentially the same code, one for residue and another for dresidue.
      // However, I think that there is a flaw in the abstraction, since I
      // cannot apparently do it with polymorphism here.

      DResidue* dr = dynamic_cast<DResidue*>(DResArray[e]);
      if (dr) {
        if (!dr->getDVal(argval, funcval, dfuncval)) {
          std::cerr << "ElementalOperation.cpp::Assemble Error in residual "
                       "computation\n";
          return false;
        }
      } else {
        std::cerr
            << "ElementalOperation.cpp::Assemble Error. Attempted to compute"
               " derivatives of a non-dresidue type\n";
        return false;
      }
    } else if (!DResArray[e]->getVal(argval, funcval)) {
      std::cerr
          << "ElementalOperation.cpp::Assemble Error in residual computation\n";
      return false;
    }

#ifdef DEBUG
    std::cout << "Assemble:: element " << e << std::endl;
    for (size_t f = 0; f < DPF.size(); ++f) {
      printIter(std::cout, funcval[f].begin(), funcval[f].end());
    }
#endif

    double* resvals  = new double[localsize];
    size_t* indices  = new size_t[localsize];
    double* dresvals = nullptr;

    if (mode == DRESIDUE) {
      dresvals = new double[localsize * localsize];
    }

    for (size_t f = 0, i = 0, j = 0; f < DPF.size(); f++) {
      for (size_t a = 0; a < DResArray[e]->getFieldDof(f); a++, i++) {
        resvals[i] = funcval[f][a];
        indices[i] = L2G.map(f, a, e);

        if (mode == DRESIDUE)
          for (size_t g = 0; g < DPF.size(); g++)
            for (size_t b = 0; b < DResArray[e]->getFieldDof(g); b++, j++)
              dresvals[j] = dfuncval[f][a][g][b];
      }
    }

    // signature (Vec, size_of_indices, size_t indices[], double[] vals, Mode)
    // VecSetValues(*ResVec, localsize, indices, resvals, ADD_VALUES);
    for (size_t i = 0; i < localsize; ++i) {
      ResVec[indices[i]] += resvals[i];
    }

    if (mode == DRESIDUE) {
      // signature (Mat, nrows_of_indices, size_t row_indices[],
      // ncols_of_indices, size_t col_indices[],
      //       double vals[], Mode)
      // algo
      // for i in 0..nrows {
      //   for j in 0..ncols {
      //     Mat[row_indices[i]][col_indices[j] += or = vals[ncols * i + j];

      // MatSetValues(*DResMat, localsize, indices, localsize, indices,
      // dresvals, ADD_VALUES);
      for (size_t i = 0; i < localsize; ++i) {
        for (size_t j = 0; j < localsize; ++j) {
          DResMat[indices[i]][indices[j]] += dresvals[localsize * i + j];
        }
      }
    }

    delete[] resvals;
    delete[] indices;

    if (mode == DRESIDUE) {
      delete[] dresvals;
    }
  }

  // VecRestoreArray(Dofs, &GDofs);

  return true;
}
bool Residue::assemble(std::vector<Residue*>& ResArray,
                       const LocalToGlobalMap& L2G, const VecDouble& Dofs,
                       VecDouble& ResVec) {
  MatDouble d;
  return _Assemble<Residue>(ResArray, L2G, Dofs, ResVec, d, RESIDUE);
}

bool DResidue::assemble(std::vector<DResidue*>& DResArray,
                        const LocalToGlobalMap& L2G, const VecDouble& Dofs,
                        VecDouble& ResVec, MatDouble& DResMat) {
  return _Assemble<DResidue>(DResArray, L2G, Dofs, ResVec, DResMat, DRESIDUE);
}
