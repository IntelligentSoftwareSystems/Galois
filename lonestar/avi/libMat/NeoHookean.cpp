/*
 * NeoHookean.cpp
 * DG++
 *
 * Created by Adrian Lew on 10/24/06.
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

#include "Material.h"

// calling default constructor
Galois::Substrate::PerThreadStorage<NeoHookean::NeoHookenTmpVec> NeoHookean::perCPUtmpVec;

static double matlib_determinant(const double *A) {
  double det;

  det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);

  return det;
}

static double matlib_inverse(const double *A, double *Ainv) {
  double det, detinv;

  det = matlib_determinant(A);
  if (fabs(det) < SimpleMaterial::DET_MIN) {
    return 0.e0;
  }

  detinv = 1. / det;
  Ainv[0] = detinv * (A[4] * A[8] - A[5] * A[7]);
  Ainv[1] = detinv * (-A[1] * A[8] + A[2] * A[7]);
  Ainv[2] = detinv * (A[1] * A[5] - A[2] * A[4]);
  Ainv[3] = detinv * (-A[3] * A[8] + A[5] * A[6]);
  Ainv[4] = detinv * (A[0] * A[8] - A[2] * A[6]);
  Ainv[5] = detinv * (-A[0] * A[5] + A[2] * A[3]);
  Ainv[6] = detinv * (A[3] * A[7] - A[4] * A[6]);
  Ainv[7] = detinv * (-A[0] * A[7] + A[1] * A[6]);
  Ainv[8] = detinv * (A[0] * A[4] - A[1] * A[3]);

  return det;
}

static void matlib_mults(const double *A, const double *B, double *C) {
  C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
  C[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
  C[2] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
  C[3] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
  C[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
  C[5] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];
  C[6] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
  C[7] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
  C[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
}

bool NeoHookean::getConstitutiveResponse(const VecDouble& strain, VecDouble& stress, VecDouble& tangents
    , const ConstRespMode& mode) const {
  // XXX: (amber) replaced unknown 3's with NDM, 9 & 81 with MAT_SIZE & MAT_SIZE ^ 2

  size_t i;
  size_t j;
  size_t k;
  size_t l;
  size_t m;
  size_t n;
  size_t ij;
  size_t jj;
  size_t kl;
  size_t jk;
  size_t il;
  size_t ik;
  size_t im;
  size_t jl;
  size_t kj;
  size_t kn;
  size_t mj;
  size_t nl;
  size_t ijkl;
  size_t indx;

  double coef;
  double defVol;
  double detC;
  double p;
  // double trace;

  NeoHookenTmpVec& tmpVec = *perCPUtmpVec.getLocal ();
  double* F = tmpVec.F;
  // double* Finv = tmpVec.Finv;
  double* C = tmpVec.C;
  double* Cinv = tmpVec.Cinv;
  double* S = tmpVec.S;
  double* M = tmpVec.M;

  
  // double detF;

  size_t J;

  std::copy (I_MAT, I_MAT + MAT_SIZE, F);

  /*Fill in the deformation gradient*/
  for (i = 0; i < NDF; i++) {
    for (J = 0; J < NDM; J++) {
      F[i * NDM + J] = strain[i * NDM + J];
    }
  }

  /* compute right Cauchy-Green tensor C */
  matlib_mults(F, F, C);

  /* compute PK2 stresses and derivatives wrt C*/
  detC = matlib_inverse(C, Cinv);
  // detF = matlib_inverse(F, Finv);

  if (detC < DET_MIN) {
    std::cerr << "NeoHookean::GetConstitutiveResponse:  close to negative jacobian\n";
    return false;
  }

  defVol = 0.5 * log(detC);
  p = Lambda * defVol;

  // trace = C[0] + C[4] + C[8];

  coef = p - Mu;

  for (j = 0, ij = 0, jj = 0; j < NDF; j++, jj += NDF + 1) {
    for (i = 0; i < NDM; i++, ij++) {
      S[ij] = coef * Cinv[ij];
    }
    S[jj] += Mu;
  }

  if (mode == COMPUTE_TANGENTS) {
    coef = Mu - p;
    for (l = 0, kl = 0, ijkl = 0; l < NDM; l++) {
      for (k = 0, jk = 0; k < NDM; k++, kl++) {
        for (j = 0, ij = 0, jl = l * NDM; j < NDM; j++, jk++, jl++) {
          for (i = 0, ik = k * NDM, il = l * NDM; i < NDM; i++, ij++, ik++, il++, ijkl++) {
            M[ijkl] = Lambda * Cinv[ij] * Cinv[kl] + coef * (Cinv[ik] * Cinv[jl] + Cinv[il] * Cinv[jk]);
          }
        }
      }
    }
  }

  if (stress.size() != MAT_SIZE) {
    stress.resize(MAT_SIZE);
  }

  /* PK2 -> PK1 */
  for (j = 0, ij = 0; j < NDM; j++) {
    for (i = 0; i < NDM; i++, ij++) {
      stress[ij] = 0.e0;
      for (k = 0, ik = i, kj = j * NDM; k < NDM; k++, ik += NDM, kj++) {
        stress[ij] += F[ik] * S[kj];
      }
    }
  }

  if (mode == COMPUTE_TANGENTS) {
    if (tangents.size() != MAT_SIZE * MAT_SIZE) {
      tangents.resize(MAT_SIZE * MAT_SIZE);
    }

    /* apply partial push-forward and add geometrical term */
    for (l = 0, ijkl = 0; l < NDM; l++) {
      for (k = 0; k < NDF; k++) {
        for (j = 0, jl = l * NDF; j < NDM; j++, jl++) {
          for (i = 0; i < NDF; i++, ijkl++) {

            tangents[ijkl] = 0.e0;

            /* push-forward */
            for (n = 0, kn = k, nl = l * NDF; n < NDM; n++, kn += NDM, nl++) {
              indx = nl * MAT_SIZE;
              for (m = 0, im = i, mj = j * NDM; m < NDM; m++, im += NDM, mj++) {
                tangents[ijkl] += F[im] * M[mj + indx] * F[kn];
              }
            }

            /* geometrical term */
            if (i == k) {
              tangents[ijkl] += S[jl];
            }

          }
        }
      }
    }
  }

  return true;
}
