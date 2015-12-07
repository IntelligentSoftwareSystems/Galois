/**
 * Linear.h: a set of linear shape functions
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

#ifndef LINEAR
#define LINEAR

#include "Shape.h"

/**
 \brief set of linear shape functions in SPD dimensions

 The linear shape functions in SPD dimensions are precisely the barycentric
 coordinates of a point in the simplex defined by SPD+1 points.

 Since quadrature points are often expressed in baricentric coordinates, this class was
 implemented to take the SPD+1 barycentric coordinates of a point as arguments only.

 Since barycentric coordinates are linearly dependent, the class chooses SPD coordinates
 as arguments of the shape functions. This choice is made by shuffling the coordinates so that the first
 SPD coordinates are the linearly independent ones.

 The possibility of shuffling becomes useful, for example, in tetrahedra. Without shuffling the class
 would have
 \f[ 
 N_a(\lambda_0,\lambda_1,\lambda_2), \qquad a=0,1,2,3 
 \f]
 and the derivatives of the shape function with respect to the same arguments. With a shuffling  of the form
 \f$(\lambda_0,\lambda_1,\lambda_2,\lambda_3)\f$ to \f$(\lambda_0,\lambda_1,\lambda_3,\lambda_2)\f$ it is
 possible to have
 \f[
 N_a(\lambda_0,\lambda_1,\lambda_3), \qquad a=0,1,2,3.
 \f]
 The derivatives returned here, are clearly different that in the previous case. The reason that this is
 useful is that these derivatives are precisely the derivatives with respect to the Cartesian coordinates
 of the shape functions in a standard parametric tet ( with vertices at (1,0,0), (0,1,0), (0,0,0), (0,0,1) )
 */

template <size_t SPD>
class Linear: public Shape {
public:
  //! Constructor \n
  //! \param iMap Shuffle of the barycentric coordinates. iMap[a] returns the position of the original 
  //! a-th barycentric coordinate after shuffling.
  //! If not provided, an identity mapping is assumed iMap[a]=a

  //! \warning No way to know if iMap has the proper length.
  Linear (const size_t * iMap = 0);

  inline virtual ~Linear () {}

  Linear (const Linear<SPD> &);

  virtual inline Linear<SPD> * clone () const {
    return new Linear<SPD> (*this);
  }

  // Accessors/Mutators
  inline size_t getNumFunctions () const {
    return SPD + 1;
  }
  inline size_t getNumVariables () const {
    return SPD;
  }

  // Functionality

  //! @param a shape function number
  //! @param x first SPD barycentric coordinates of the point
  //! \warning Does not check range for parameter a
  double getVal (size_t a, const double *x) const;
  //! @param a shape function number
  //! @param x first spartial_dimension barycentric coordinates of the point
  //! @param i partial derivative number 
  //! Returns derivative with respect to the barycentric coordinates
  //! \warning Does not check range for parameters a and i
  double getDVal (size_t a, const double *x, size_t i) const;

private:
  size_t bctMap[SPD + 1];
};

template<size_t SPD>
Linear<SPD>::Linear (const size_t * iMap) {
  for (size_t a = 0; a < SPD + 1; a++)
    bctMap[a] = a;

  if (iMap != 0) {
    for (size_t a = 0; a < SPD + 1; a++) {
      bctMap[a] = iMap[a];
    }
  }

  return;
}

template<size_t SPD>
Linear<SPD>::Linear (const Linear<SPD> &Lin) {
  for (size_t a = 0; a < SPD + 1; a++) {
    bctMap[a] = Lin.bctMap[a];
  }
}

template<size_t SPD>
double Linear<SPD>::getVal (size_t a, const double *x) const {
  if (bctMap[a] != SPD) {
    return x[bctMap[a]];
  }
  else {
    double va = 0;

    for (size_t k = 0; k < SPD; k++) {
      va += x[k];
    }

    return 1 - va;
  }
}

template<size_t SPD>
double Linear<SPD>::getDVal (size_t a, const double *x, size_t i) const {
  if (bctMap[a] != SPD) {
    return bctMap[a] == i ? 1 : 0;
  } else {
    return -1;
  }
}

#endif
