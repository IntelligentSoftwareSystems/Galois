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
 * StressWork.h
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

#ifndef STRESSWORK
#define STRESSWORK

#include <vector>
#include <algorithm>

#include <cassert>

#include "galois/substrate/PerThreadStorage.h"

#include "ElementalOperation.h"
#include "AuxDefs.h"

/**
 \brief Computes the virtual work of the stress tensor, and its derivative

 The virtual work of the stress tensor \f${\bf P}\f$ is defined as
 \f[
 \int_{E} P_{iJ} v_{i,J}\ d\Omega,
 \f]
 where \f${\bf v}\f$ is a virtual displacement field. This operation works
 for two and three-dimensional problems. In two-dimensional
 problems plane strain is assumed, i.e., the displacements and virtual
 displacements have the form \f$(v_1(x_1,x_2), v_2(x_1,x_2), 0)\f$.

 StressWork works only on SolidElements, since it needs a SimpleMaterial
 to compute the stress tensor.

 StressWork computes the residue
 \f[
 R[f][a] = \int_{E} P_{fJ} N_{a,J}\ d\Omega,
 \f]
 where \f$N_{a,f}\f$ is the derivative of shape function associated to
 degree of freedom \f$a\f$ in direction \f$f\f$.

 The derivative of this residue is
 \f[
 DR[f][a][g][b] = \int_{E} A_{fJgL} N_{a,J} N_{b,L}\ d\Omega,
 \f]
 where \f$A\f$ are the elastic moduli \f$\partial{\bf P}/\partial {\bf F}\f$.
 */
class StressWork : public DResidue {
protected:
  enum GetValMode {
    VAL,
    DVAL,
  };

  //! \warning argval should contain displacements, not deformation mapping
  bool getDValIntern(const MatDouble& argval, MatDouble& funcval,
                     FourDVecDouble& dfuncval, const GetValMode& mode) const;

private:
  /**
   * Contains the temporary vectors used by @see getDValIntern
   * Instead of creating and destroying new vectors on every call,
   * which happens at least once per iteration, we reuse vectors
   * from this struct. There's one instance per thread of this struct
   */
  struct StressWorkTmpVec {
    static const size_t MAT_SIZE = SimpleMaterial::MAT_SIZE;

    VecSize_t nDof;
    VecSize_t nDiv;

    MatDouble DShape;
    MatDouble IntWeights;

    VecDouble A;
    VecDouble F;
    VecDouble P;

    StressWorkTmpVec()
        : A(MAT_SIZE * MAT_SIZE, 0.0), F(MAT_SIZE, 0.0), P(MAT_SIZE, 0.0) {}

    void adjustSizes(size_t Dim) {
      if (nDof.size() != Dim) {
        nDof.resize(Dim);
      }

      if (nDiv.size() != Dim) {
        nDiv.resize(Dim);
      }

      if (DShape.size() != Dim) {
        DShape.resize(Dim);
      }

      if (IntWeights.size() != Dim) {
        IntWeights.resize(Dim);
      }

      if (A.size() != MAT_SIZE * MAT_SIZE) {
        A.resize(MAT_SIZE * MAT_SIZE, 0.0);
      }

      if (F.size() != MAT_SIZE) {
        F.resize(MAT_SIZE, 0.0);
      }

      if (P.size() != MAT_SIZE) {
        P.resize(MAT_SIZE, 0.0);
      }
    }
  };

  /**
   * Per thread storage for temporary vectors used in @see getDValIntern
   */
  typedef galois::substrate::PerThreadStorage<StressWorkTmpVec*> PerCPUtmpVecTy;

  static PerCPUtmpVecTy perCPUtmpVec;

public:
  //! Construct a StressWork object with fields "field1, field2 and field3" as
  //! the three dimensional displacement fields.
  //! @param IElm pointer to the element over which the value will be computed.
  //! The Input object is non-const, since these can be modified during the
  //! operation.  The object pointed to is not destroyed when the operation is.
  //! @param SM SimpleMaterial object used to compute the stress and moduli. It
  //! is only referenced, not copied.
  //! @param fieldsUsed vector containing ids of fields being computed starting
  //! with 0 Cartesian component  of the displacement field. If not provided, it
  //! is assumed that it is a plane strain case.
  StressWork(const Element& IElm, const SimpleMaterial& SM,
             const VecSize_t& fieldsUsed)
      : DResidue(IElm, SM, fieldsUsed) {

    assert(fieldsUsed.size() > 0 && fieldsUsed.size() <= 3);
  }

  virtual ~StressWork() {}

  StressWork(const StressWork& SW) : DResidue(SW) {}

  virtual StressWork* clone() const { return new StressWork(*this); }

  VecDouble getIntegrationWeights(size_t fieldnumber) const {
    return BaseResidue::element.getIntegrationWeights(fieldnumber);
  }

  //! \warning argval should contain displacements, not deformation mapping
  bool getDVal(const MatDouble& argval, MatDouble& funcval,
               FourDVecDouble& dfuncval) const {
    return getDValIntern(argval, funcval, dfuncval, DVAL);
  }

  //! \warning argval should contain displacements, not deformation mapping
  bool getVal(const MatDouble& argval, MatDouble& funcval) const {
    FourDVecDouble d;
    return getDValIntern(argval, funcval, d, VAL);
  }

private:
  static void copyVecDouble(const VecDouble& vin, VecDouble& vout) {
    if (vout.size() != vin.size()) {
      vout.resize(vin.size());
    }

    std::copy(vin.begin(), vin.end(), vout.begin());
  }
};

#endif
