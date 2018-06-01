/* Sriramajayam */

// testP13DTrace.cpp

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

  P13DTrace<2> Trace1(Elm, P13DTrace<2>::FaceOne, P13DTrace<2>::ThreeDofs);
  P13DTrace<2> Trace2(Elm, P13DTrace<2>::FaceTwo, P13DTrace<2>::ThreeDofs);
  P13DTrace<2> Trace3(Elm, P13DTrace<2>::FaceThree, P13DTrace<2>::ThreeDofs);
  P13DTrace<2> Trace4(Elm, P13DTrace<2>::FaceFour, P13DTrace<2>::ThreeDofs);

  P13DTrace<2> * Faces[] = { &Trace1, &Trace2, &Trace3, &Trace4};

  for(int i=1; i<2; i++) // Change to test different/all traces.
    {
      std::cout<<"\n Testing Face: "<<i<<".\n";
      std::cout<<"\n Number of Fields: "<<Faces[i]->GetFields()<<" should be 2\n";
      std::cout<<"\nNumber of dof field(0): "<<Faces[i]->getDof(0)<<" should be 3\n";
      std::cout<<"\nNumber of dof field(1): "<<Faces[i]->getDof(1)<<" should be 3\n";


      // Printing Shape functions at quadrature points.
      for(int f=0; f<2; f++)
	{
	  std::cout <<"\n Shape Function values at quad point for field "<< f<< ":\n";
	  for(unsigned int q=0; q<Faces[i]->getShapes(f).size(); q++)
	    std::cout << Faces[i]->getShapes(f)[q] <<" ";
	  
	  std::cout << "\n";
	}

      // Printing integration weights at quad points:
       for(int f=0; f<2; f++)
	{
	  std::cout <<"\n Integration weights at quad point for field "<< f<< ":\n";
	  for(unsigned int q=0; q<Faces[i]->getIntegrationWeights(f).size(); q++)
	    std::cout << Faces[i]->getIntegrationWeights(f)[q] <<" ";
	  
	  std::cout << "\n";
	}

        // Printing integration quad points:
       for(int f=0; f<2; f++)
	{
	  std::cout <<"\n Quad point coordinates for field "<< f<< ":\n";
	  for(unsigned int q=0; q<Faces[i]->getIntegrationPtCoords(f).size(); q++)
	    std::cout << Faces[i]->getIntegrationPtCoords(f)[q] <<" ";
	  
	  std::cout << "\n";
	}

       std::cout<<"\n Shape function derivatives not tested. \n";
    }

  std::cout<< "\n Test Successful. \n\n";

}
       
       


