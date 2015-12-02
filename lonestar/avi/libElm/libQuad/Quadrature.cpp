/*
 * Quadrature.cpp
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

#include <iostream>
#include "Quadrature.h"
#include <cmath>

Quadrature::Quadrature (const double * const xqdat, const double * const wqdat, const size_t NC,
    const size_t NQ) :
  numMapCoordinates (NC), numShapeCoordinates (NC), numQuadraturePoints (NQ) {
#if 0
  if (NQ < 0 || NC < 0) {
    std::cerr << "Quadrature::Quadrature: Negative number of quadrature points or coordinates\n";
    exit (1); // Bad..., to be improved in the future with exceptions
  }
#endif

  xqshape = xqmap = new double[NQ * NC];
  wq = new double[NQ];

  for (size_t q = 0; q < NQ; q++) {
    for (size_t i = 0; i < NC; i++) {
      xqmap[q * NC + i] = xqdat[q * NC + i];
    }
    wq[q] = wqdat[q];
  }
}

Quadrature::Quadrature (const double * const xqdatmap, const double * const xqdatshape,
    const double * const wqdat, const size_t NCmap, const size_t NCshape, const size_t NQ) :
  numMapCoordinates (NCmap), numShapeCoordinates (NCshape), numQuadraturePoints (NQ) {
#if 0
  if (NQ < 0 || NCmap < 0 || NCshape < 0) {
    std::cerr << "Quadrature::Quadrature: Negative number of quadrature points or coordinates\n";
    exit (1); // Bad..., to be improved in the future with exceptions
  }
#endif

  xqmap = new double[NQ * NCmap];
  xqshape = new double[NQ * NCshape];
  wq = new double[NQ];

  for (size_t q = 0; q < NQ; q++) {
    for (size_t i = 0; i < NCmap; i++) {
      xqmap[q * NCmap + i] = xqdatmap[q * NCmap + i];
    }
    for (size_t i = 0; i < NCshape; i++) {
      xqshape[q * NCshape + i] = xqdatshape[q * NCshape + i];
    }
    wq[q] = wqdat[q];
  }
}

Quadrature::Quadrature (const Quadrature &SQ) :
  numMapCoordinates (SQ.numMapCoordinates),numShapeCoordinates (SQ.numShapeCoordinates),
      numQuadraturePoints (SQ.numQuadraturePoints) {
  xqmap = new double[numMapCoordinates * numQuadraturePoints];
  if (SQ.xqmap != SQ.xqshape)
    xqshape = new double[numShapeCoordinates * numQuadraturePoints];
  else
    xqshape = xqmap;
  wq = new double[numQuadraturePoints];

  for (size_t q = 0; q < numQuadraturePoints; q++) {
    for (size_t i = 0; i < numMapCoordinates; i++)
      xqmap[q * numMapCoordinates + i] = SQ.xqmap[q * numMapCoordinates + i];
    if (xqmap != xqshape)
      for (size_t i = 0; i < numShapeCoordinates; i++)
        xqshape[q * numShapeCoordinates + i] = SQ.xqshape[q * numShapeCoordinates + i];

    wq[q] = SQ.wq[q];
  }
}

Quadrature * Quadrature::clone () const {
  return new Quadrature (*this);
}

// Build specific quadratures  

// 3-point quadrature on Triangle (0,0), (1,0), (0,1)
const double Triangle_1::BulkCoordinates[] = 
  {0.6666666666666667e0,0.1666666666666667e0,
   0.1666666666666667e0,0.6666666666666667e0,
   0.1666666666666667e0,0.1666666666666667e0};
const double Triangle_1::BulkWeights []    =    {1./6.,1./6.,1./6.};

const double Triangle_1::FaceMapCoordinates[] = 
  {0.5 + 0.577350269/2., 0.5 - 0.577350269/2.}; // First barycentric coordinate in reference segment (0,1)

const double Triangle_1::FaceOneShapeCoordinates[] = 
  {0.5 + 0.577350269/2., 0.5 - 0.577350269/2.,
   0.5 - 0.577350269/2., 0.5 + 0.577350269/2.};  // Coordinates in the reference triangle 
const double Triangle_1::FaceOneWeights []    =    {1./2.,1./2.};

const double Triangle_1::FaceTwoShapeCoordinates[] = 
  {0., 0.5 + 0.577350269/2.,
   0., 0.5 - 0.577350269/2.};                    // Coordinates in the reference triangle   
const double Triangle_1::FaceTwoWeights [] = {1./2.,1./2.};

const double Triangle_1::FaceThreeShapeCoordinates[] = 
  {0.5 - 0.577350269/2., 0., 
   0.5 + 0.577350269/2., 0.};                    // Coordinates in the reference triangle 
const double Triangle_1::FaceThreeWeights [] =  {1./2.,1./2.};

const Quadrature * const Triangle_1::Bulk = 
  new Quadrature(Triangle_1::BulkCoordinates, 
		 Triangle_1::BulkWeights, 2, 3);
const Quadrature * const Triangle_1::FaceOne =
  new Quadrature(Triangle_1::FaceMapCoordinates, 
		 Triangle_1::FaceOneShapeCoordinates, 
		 Triangle_1::FaceOneWeights, 1, 2, 2);
const Quadrature * const Triangle_1::FaceTwo = 
  new Quadrature(Triangle_1::FaceMapCoordinates, 
		 Triangle_1::FaceTwoShapeCoordinates, 
		 Triangle_1::FaceTwoWeights, 1, 2, 2);
const Quadrature * const Triangle_1::FaceThree = 
  new Quadrature(Triangle_1::FaceMapCoordinates, 
		 Triangle_1::FaceThreeShapeCoordinates, 
		 Triangle_1::FaceThreeWeights, 1, 2, 2);


// 2-point Gauss quadrature in a Segment (0,1)
const double Line_1::BulkCoordinates[] = {0.5 + 0.577350269/2., 
					  0.5 - 0.577350269/2.};
const double Line_1::BulkWeights []    =    {1./2.,1./2.};

const Quadrature * const Line_1::Bulk = 
  new Quadrature(Line_1::BulkCoordinates, 
		 Line_1::BulkWeights, 1, 2);


// 4-point Gauss quadrature in a Tet (1,0,0), (0,1,0), (0,0,0), (0,0,1)

const double Tet_1::BulkCoordinates[] = 
  {0.58541020e0, 0.13819660e0, 0.13819660e0,
   0.13819660e0, 0.58541020e0, 0.13819660e0,
   0.13819660e0, 0.13819660e0, 0.58541020e0,
   0.13819660e0, 0.13819660e0, 0.13819660e0};

const double Tet_1::BulkWeights []    =    {1./24.,
						 1./24.,
						 1./24.,
						 1./24.};

const double Tet_1::FaceMapCoordinates[] = {2./3., 1./6.,
						 1./6., 2./3.,
						 1./6., 1./6.}; 

// Face 1 : 2-1-0.
const double Tet_1::FaceOneShapeCoordinates[] = 
  { 1./6., 1./6., 0.,
    1./6., 2./3., 0.,
    2./3., 1./6., 0.};
 
const double Tet_1::FaceOneWeights [] = { 1./6., 1./6., 1./6.};


// Face 2 : 2-0-3.
const double Tet_1::FaceTwoShapeCoordinates[] = 
  { 1./6., 0., 1./6.,
    2./3., 0., 1./6., 
    1./6., 0., 2./3.};

const double Tet_1::FaceTwoWeights [] = { 1./6., 1./6., 1./6.};


// Face 3: 2-3-1.
const double Tet_1::FaceThreeShapeCoordinates[] = 
  { 0., 1./6., 1./6.,
    0., 1./6., 2./3.,
    0., 2./3., 1./6.};

const double Tet_1::FaceThreeWeights [] = { 1./6., 1./6., 1./6.};


// Face 4: 0-1-3.
const double Tet_1::FaceFourShapeCoordinates [] =
  { 2./3., 1./6., 1./6.,
    1./6., 2./3., 1./6.,
    1./6., 1./6., 2./3.};

const double Tet_1::FaceFourWeights [] = { 1./6., 1./6., 1./6.};


const Quadrature * const Tet_1::Bulk = 
  new Quadrature(Tet_1::BulkCoordinates, 
		 Tet_1::BulkWeights, 3, 4);

const Quadrature * const Tet_1::FaceOne =
  new Quadrature(Tet_1::FaceMapCoordinates, 
		 Tet_1::FaceOneShapeCoordinates, 
		 Tet_1::FaceOneWeights, 2, 3, 3);

const Quadrature * const Tet_1::FaceTwo = 
  new Quadrature(Tet_1::FaceMapCoordinates, 
		 Tet_1::FaceTwoShapeCoordinates, 
		 Tet_1::FaceTwoWeights, 2, 3, 3);

const Quadrature * const Tet_1::FaceThree = 
  new Quadrature(Tet_1::FaceMapCoordinates, 
		 Tet_1::FaceThreeShapeCoordinates, 
		 Tet_1::FaceThreeWeights, 2, 3, 3);

const Quadrature * const Tet_1::FaceFour = 
  new Quadrature(Tet_1::FaceMapCoordinates, 
		 Tet_1::FaceFourShapeCoordinates, 
		 Tet_1::FaceFourWeights, 2, 3, 3);

