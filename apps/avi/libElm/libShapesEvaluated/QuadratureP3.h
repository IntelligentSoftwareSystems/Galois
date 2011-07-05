#include "Quadrature.h"

#ifndef QUADRATUREP3
#define QUADRATUREP3

/*!
 * \brief Class for 4 point quadrature rules for tetrahedrons.
 *
 * 4-point Gauss quadrature coordinates in the tetrahedron with 
 * 0(1,0,0), 1(0,1,0), 2(0,0,0), 3(0,0,1) as vertices.
 * Barycentric coordinates are used for the Gauss points.
 * Barycentric coordinates are specified with respect to vertices 1,2 and 4
 * in that order.Coordinate of vertex 3 is not independent.
 *
 * Quadrature for Faces:
 * Faces are ordered as - 
 * Face 1: 2-1-0,
 * Face 2: 2-0-3,
 * Face 3: 2-3-1,
 * Face 4: 0-1-3.
 */

class Tet_4Point: public SpecificQuadratures
{
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;
  
  //! Face (2-1-0) quadrature
  static const Quadrature* const FaceOne;
  //! Face (2-0-3) quadrature
  static const Quadrature* const FaceTwo;
  //! Face (2-3-1) quadrature
  static const Quadrature* const FaceThree;
  //! Face (0-1-3) quadrature
  static const Quadrature* const FaceFour;
  
private:
  static const double BulkCoordinates[];
  static const double BulkWeights[];
  static const double FaceMapCoordinates[];
  static const double FaceOneShapeCoordinates[];
  static const double FaceOneWeights[];
  static const double FaceTwoShapeCoordinates[];
  static const double FaceTwoWeights[];
  static const double FaceThreeShapeCoordinates[];
  static const double FaceThreeWeights[];
  static const double FaceFourShapeCoordinates[];
  static const double FaceFourWeights[];
};



//! \brief 11 point quadrature rule for tetrahedron
//! Degree of precision 4, number of points 11.
class Tet_11Point: public SpecificQuadratures
{
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;
  
  //! \todo Include face quadrature rules if needed.
  
private:
  static const double BulkCoordinates[];
  static const double BulkWeights[];
};

//! \brief 15 point quadrature rule for tetrahedron
//! Degree of precision 5, number of points 15.
class Tet_15Point: public SpecificQuadratures
{
public:
  //! Bulk quadrature
  static const Quadrature* const Bulk;
  
  //! \todo Include face quadrature rules if needed.
  
private:
  static const double BulkCoordinates[];
  static const double BulkWeights[];
};

#endif
// Sriramajayam

