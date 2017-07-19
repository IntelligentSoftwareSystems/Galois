/* Sriramajayam */

// testP13DElementBoundaryTraces.cpp.

#include "P13DElement.h"
#include <iostream>

int main()
{
  double V0[] = {1,0,0,
		 0,1,0,
		 0,0,0,
		 0,0,1};
  
  VecDouble Vertices(V0, V0+12);
  
  Tetrahedron::SetGlobalCoordinatesArray(Vertices);
  Triangle<3>::SetGlobalCoordinatesArray(Vertices);
  Segment<3>::SetGlobalCoordinatesArray(Vertices);
  
  P13DElement<2> Elm(1,2,3,4);

  P13DElementBoundaryTraces<2>
    EBT(Elm, true, false, true, true, P13DTrace<2>::ThreeDofs);

  std::cout<<"\n Number of Traces: "<<EBT.getNumTraceFaces()<<" should be 3. \n";

  for(unsigned int a=0; a<3; a++)  // Change to test different/all traces.
    {
      int facenumber = EBT.getTraceFaceIds()[a];
      std::cout<<" \n FaceNumber :"<<facenumber<<"\n";
      
      std::cout<<"\n Normal to face: ";
      for(unsigned int q=0; q<EBT.getNormal(a).size(); q++)
	std::cout<< EBT.getNormal(a)[q]<<", ";
      std::cout<<"\n";

    }

  for(unsigned int a=0; a<1; a++)
    {
      int facenumber = EBT.getTraceFaceIds()[a];
      std::cout<<"\nFace Number :"<<facenumber<<"\n";
      
      const Element &face = EBT[a];

      // Integration weights:
      std::cout<<"\n Integration weights: \n";
      for(unsigned int q=0; q<face.getIntegrationWeights(0).size(); q++)
	std::cout<<face.getIntegrationWeights(0)[q]<<", ";
    
       // Integration points:
      std::cout<<"\n Integration point coordinates: \n";
      for(unsigned int q=0; q<face.getIntegrationPtCoords(0).size(); q++)
	std::cout<<face.getIntegrationPtCoords(0)[q]<<", ";
      

      // Shape functions:
      std::cout<<"\n Shape Functions at quadrature points: \n";
      for(unsigned int q=0; q<face.getShapes(0).size(); q++)
	std::cout<<face.getShapes(0)[q]<<", ";
      
      std::cout<<"\n";
    }

}
 
