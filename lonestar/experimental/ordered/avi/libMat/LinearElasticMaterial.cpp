/*
 * LinearElasticMaterial.cpp
 * DG++
 *
 * Created by Adrian Lew on 11/19/06.
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
#include <cmath>
#include <iostream>

// XXX: (amber) replaced 3,9. 27, 81 with NDM, MAT_SIZE
bool LinearElasticBase::getConstitutiveResponse(
    const VecDouble& strain, VecDouble& stress, VecDouble& tangents,
    const ConstRespMode& mode) const {

  // Compute stress
  if (stress.size() != MAT_SIZE) {
    stress.resize(MAT_SIZE);
  }

  for (size_t i = 0; i < NDM; i++) {
    for (size_t J = 0; J < NDM; J++) {
      stress[i * NDM + J] = 0;

      for (size_t k = 0; k < NDM; k++) {
        for (size_t L = 0; L < NDM; L++) {
          stress[i * NDM + J] += getModuli(i, J, k, L) *
                                 (strain[k * NDM + L] - I_MAT[k * NDM + L]);
        }
      }
    }
  }

  // Compute tangents, if needed
  if (mode == COMPUTE_TANGENTS) {

    if (tangents.size() != MAT_SIZE * MAT_SIZE) {
      tangents.resize(MAT_SIZE * MAT_SIZE);
    }

    for (size_t i = 0; i < NDM; i++)
      for (size_t J = 0; J < NDM; J++)
        for (size_t k = 0; k < NDM; k++)
          for (size_t L = 0; L < NDM; L++)
            tangents[i * MAT_SIZE * NDM + J * MAT_SIZE + k * NDM + L] =
                getModuli(i, J, k, L);
  }

  return true;
}
