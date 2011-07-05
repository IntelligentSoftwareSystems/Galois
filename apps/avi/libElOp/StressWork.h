/*
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

#include <cassert>

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
class StressWork:  public BaseResidue {
public:
  //! Construct a StressWork object with fields "field1, field2 and field3" as 
  //! the three dimensional displacement fields. 
  //! @param IElm pointer to the element over which the value will be computed. 
  //! The Input object is non-const, since these can be modified during the 
  //! operation.  The object pointed to is not destroyed when the operation is.
  //! @param SM SimpleMaterial object used to compute the stress and moduli. It is
  //! only referenced, not copied.
  //! @param field1 field number in the Element that represents the first 
  //! Cartesian component  of the displacement field
  //! @param field2 field number in the Element that represents the second
  //! Cartesian component  of the displacement field
  //! @param field3 field number in the Element that represents the third
  //! Cartesian component  of the displacement field. If not provided, it is
  //! assumed that it is a plane strain case.
  StressWork(const Element& IElm, const SimpleMaterial &SM, const std::vector<size_t>& fieldsUsed)
    : BaseResidue (IElm, SM, fieldsUsed) {
    assert (fieldsUsed.size() > 0 && fieldsUsed.size () <= 3);
  }

  virtual ~StressWork() {
  }


  StressWork(const StressWork & SW) : BaseResidue (SW) {
  }


  virtual StressWork * clone() const {
    return new StressWork(*this);
  }


  VecDouble getIntegrationWeights(size_t fieldnumber) const {
    return BaseResidue::element.getIntegrationWeights(fieldnumber);
  }

  //! \warning argval should contain displacements, not deformation mapping
  bool getDVal(const MatDouble &argval, MatDouble * funcval,
      FourDVecDouble * dfuncval) const;

  //! \warning argval should contain displacements, not deformation mapping
  bool getVal(const MatDouble &argval, MatDouble * funcval) const {
    return getDVal(argval, funcval, 0);
  }

};

#endif
