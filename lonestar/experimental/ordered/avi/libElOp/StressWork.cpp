/*
 * StressWork.cpp
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
#include <iostream>

#include "StressWork.h"
#include "galois/gIO.h"

StressWork::PerCPUtmpVecTy StressWork::perCPUtmpVec;

bool StressWork::getDValIntern(const MatDouble& argval, MatDouble& funcval,
                               FourDVecDouble& dfuncval,
                               const GetValMode& mode) const {
  const size_t Dim = fieldsUsed.size();

  // XXX: (amber) replaced 3 with NDM, 9 with MAT_SIZE and so on ...
  const size_t MAT_SIZE = SimpleMaterial::MAT_SIZE;
  const size_t NDM      = SimpleMaterial::NDM;

  // We should have the same quadrature points in  all fields used
  size_t nquad = element.getIntegrationWeights(fieldsUsed[0]).size();

  StressWorkTmpVec*& tmpVecPtr = *StressWork::perCPUtmpVec.getLocal();

  if (!tmpVecPtr) {
    // TODO: memory not freed.
    tmpVecPtr = new StressWorkTmpVec;
  }

  StressWorkTmpVec& tmpVec = *tmpVecPtr;
  tmpVec.adjustSizes(Dim);

  VecSize_t& nDof = tmpVec.nDof;
  VecSize_t& nDiv = tmpVec.nDiv;

  MatDouble& DShape     = tmpVec.DShape;
  MatDouble& IntWeights = tmpVec.IntWeights;

  VecDouble& A = tmpVec.A;
  VecDouble& F = tmpVec.F;
  VecDouble& P = tmpVec.P;

  for (size_t f = 0; f < Dim; ++f) {
    nDof[f] = element.getDof(fieldsUsed[f]);
    nDiv[f] = element.getNumDerivatives(fieldsUsed[f]);

    // DShape[f] = element.getDShapes(fieldsUsed[f]);
    // do copy instead of assignment to pervent calls to allocator
    StressWork::copyVecDouble(element.getDShapes(fieldsUsed[f]), DShape[f]);

    // IntWeights[f] = element.getIntegrationWeights(fieldsUsed[f]);
    // do copy instead of assignment to pervent calls to allocator
    StressWork::copyVecDouble(element.getIntegrationWeights(fieldsUsed[f]),
                              IntWeights[f]);
  }

  if (funcval.size() < Dim) {
    funcval.resize(Dim);
  }

  for (size_t f = 0; f < Dim; f++) {
    if (funcval[f].size() < nDof[f]) {
      funcval[f].resize(nDof[f], 0.);
    } else {
      std::fill(funcval[f].begin(), funcval[f].end(), 0.0);
      //      for (size_t a = 0; a < nDof[f]; ++a) {
      //        funcval[f][a] = 0.;
      //      }
    }
  }

  if (mode == DVAL) {

    if (dfuncval.size() < Dim) {
      dfuncval.resize(Dim);
    }

    for (size_t f = 0; f < Dim; ++f) {
      if (dfuncval[f].size() < nDof[f]) {
        dfuncval[f].resize(nDof[f]);
      }

      for (size_t a = 0; a < nDof[f]; ++a) {
        if (dfuncval[f][a].size() < Dim) {
          dfuncval[f][a].resize(Dim);
        }

        for (size_t g = 0; g < Dim; ++g) {
          if (dfuncval[f][a][g].size() < nDof[g]) {
            dfuncval[f][a][g].resize(nDof[g], 0.);
          } else {
            for (size_t b = 0; b < nDof[g]; ++b) {
              dfuncval[f][a][g][b] = 0.;
            }
          }

        } // end for
      }
    }
  }

  for (size_t q = 0; q < nquad; ++q) {
    // Compute gradients

    //    F[0] = F[4] = F[8] = 1.;
    //    F[1] = F[2] = F[3] = F[5] = F[6] = F[7] = 0.;
    std::copy(SimpleMaterial::I_MAT, SimpleMaterial::I_MAT + MAT_SIZE,
              F.begin());

    for (size_t f = 0; f < Dim; ++f) {
      for (size_t a = 0; a < nDof[f]; ++a) {
        for (size_t J = 0; J < nDiv[f]; ++J) {
          // double t = argval[f][a] * DShape[f][q * nDof[f] * nDiv[f] + a *
          // nDiv[f] + J];
          double t =
              DShape[f][q * nDof[f] * nDiv[f] + a * nDiv[f] + J] * argval[f][a];
          F[f * NDM + J] += t;
        }
      }
    }

    if (!material.getConstitutiveResponse(F, P, A,
                                          SimpleMaterial::SKIP_TANGENTS)) {
      GALOIS_DIE("StressWork.cpp: Error in the constitutive response\n");
      return false;
    }

    for (size_t f = 0; f < Dim; ++f) {
      for (size_t a = 0; a < nDof[f]; ++a) {
        for (size_t J = 0; J < nDiv[f]; ++J) {
          funcval[f][a] += IntWeights[f][q] * P[f * NDM + J] *
                           DShape[f][q * nDof[f] * nDiv[f] + a * nDiv[f] + J];
        }
      }
    }

    if (mode == DVAL) {
      for (size_t f = 0; f < Dim; ++f) {
        for (size_t a = 0; a < nDof[f]; ++a) {
          for (size_t g = 0; g < Dim; ++g) {
            for (size_t b = 0; b < nDof[g]; ++b) {
              for (size_t J = 0; J < nDiv[f]; ++J) {
                for (size_t L = 0; L < nDiv[g]; ++L) {
                  dfuncval[f][a][g][b] +=
                      IntWeights[f][q] *
                      A[f * NDM * MAT_SIZE + J * MAT_SIZE + g * NDM + L] *
                      DShape[f][q * nDof[f] * nDiv[f] + a * nDiv[f] + J] *
                      DShape[g][q * nDof[g] * nDiv[g] + b * nDiv[g] + L];
                }
              }
            }
          }
        }
      }
    }
  }

  return true;
}
