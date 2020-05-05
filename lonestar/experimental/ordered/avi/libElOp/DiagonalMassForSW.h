/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism.
 * The code is being released under the terms of the 3-Clause
 * BSD License (a
 * copy is located in LICENSE.txt at the top-level
 * directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All
 * rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF
 * MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND
 * WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE
 * FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS
 * OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under
 * no circumstances
 * shall University be liable for incidental, special,
 * indirect, direct or
 * consequential damages or loss of profits, interruption
 * of business, or
 * related expenses which may arise from use of Software or
 * Documentation,
 * including but not limited to those resulting from defects
 * in Software and/or
 * Documentation, or loss or inaccuracy of data of any
 * kind.
 */

/**
 * DiagonalMassForSW.h
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

// Sriramajayam

#ifndef DIAGONALMASSMATRIXFORSW
#define DIAGONALMASSMATRIXFORSW

#include "AuxDefs.h"
#include "ElementalOperation.h"

//! \brief Class to compute a diagonal mass matrix for StressWork.
/** This class computes a diagonalized form of the (exact) mass matrix
 * \f$ M[f][a][b] = \int_E \rho_0 N_a^f N_b^f. \f$
 *
 * Since a diagonal mass matrix is often desired, the entries in each
 * row of the exact mass-matrix are lumped together on the diagonal
 * entry \f$ M[f][a][a] \f$.
 *
 * A mass-vector is assembled (instead of a matrix) with each entry
 * compued as \f$ M[f][a] = \int_E \rho_0 N_a^f \f$
 * where \f$a\f$ runs over all degrees of freedom for
 * field \f$ f \f$.
 *
 * Keeping this in mind, this class inherits the class Residue to
 * assemble the mass vector.  It has pointers to the element for which
 * the mass matrix is to be computed and the material provided over
 * the element. It assumes that there are a minimum of two fields over
 * the element with an optional third.  It implements the member
 * function getVal of Residue to compute the elemental contribution to
 * the mass-vector.
 *
 */

class DiagonalMassForSW : public BaseResidue {
public:
  //! Constructor
  //! \param IElm Pointer to element over which mass is to be compued.
  //! \param SM SimpleMaterial over the element.
  //! \param fieldsUsed vector containing ids of fields being computed starting
  //! with 0
  inline DiagonalMassForSW(const Element& IElm, const SimpleMaterial& SM,
                           const VecSize_t& fieldsUsed)
      : BaseResidue(IElm, SM, fieldsUsed) {
    assert(fieldsUsed.size() > 0 && fieldsUsed.size() <= 3);
  }

  //! Destructor
  virtual ~DiagonalMassForSW() {}

  //! Copy constructor
  //! \param DMM DiagonalMassForSW object to be copied.
  inline DiagonalMassForSW(const DiagonalMassForSW& DMM) : BaseResidue(DMM) {}

  //! Cloning mechanism
  virtual DiagonalMassForSW* clone() const {
    return new DiagonalMassForSW(*this);
  }

  //! Computes the elemental contribution to the mass-vector.
  //! \param argval See Residue. It is a dummy argument since integrations are
  //! done over the reference. \param funcval See Residue.
  bool getVal(const MatDouble& argval, MatDouble& funcval) const;
};

#endif
