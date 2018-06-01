#include "Tetrahedron.h"

const double Tetrahedron::ParamCoord[] = 
      {1,0,0, // Parametric coordinates of the tet.
			 0,1,0,
			 0,0,0,
			 0,0,1};
  
const size_t Tetrahedron::FaceNodes[] = 
        {2,1,0,   // face 0
		     2,0,3,   // face 1
		     2,3,1,   // face 2
		     0,1,3};  // face 3
