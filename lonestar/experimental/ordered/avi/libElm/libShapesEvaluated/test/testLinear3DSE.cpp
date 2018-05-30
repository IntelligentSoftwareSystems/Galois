// Sriramajayam

// Purpose :  To check ShapesEvaluatedP13D

#include "Tetrahedron.h"
#include "ShapesEvaluatedP13D.h"
#include <iostream>

static void PrintData(ShapesP13D::Bulk PShapes);

int main()
{
  double TempVertices0[] = {1,0,0, 
			    0,1,0, 
			    0,0,0, 
			    0,0,1};
  
  std::vector<double> Vertices0(TempVertices0,TempVertices0+12);
   
  Tetrahedron::SetGlobalCoordinatesArray(Vertices0);

  Tetrahedron P1(1,2,3,4);
  ShapesP13D::Bulk P1Shapes(P1);

  std::cout << "Parametric tet\n";
  PrintData(P1Shapes);

  std::cout << "\nTwice Parametric tet\n";

  double TempVertices1[] = {2,0,0, 
			    0,2,0, 
			    0,0,0, 
			    0,0,2};

  std::vector<double> Vertices1(TempVertices1,TempVertices1+12);
  Tetrahedron::SetGlobalCoordinatesArray(Vertices1);
  Tetrahedron P2(1,2,3,4);
  ShapesP13D::Bulk P2Shapes(P2);

  PrintData(P2Shapes);
  
}




static void PrintData(ShapesP13D::Bulk PShapes)
{
  std::cout << "Function values\n";
  for(unsigned int a=0; a<PShapes.getShapes().size(); a++)
    std::cout <<  PShapes.getShapes()[a] << " ";
  std::cout << "\n";
  
  std::cout << "Function derivative values\n";
  for(unsigned int a=0; a<PShapes.getDShapes().size(); a++)
    std::cout <<  PShapes.getDShapes()[a] << " ";
  std::cout << "\n";
  
  std::cout << "Integration weights\n";
  for(unsigned int a=0; a<PShapes.getIntegrationWeights().size(); a++)
    std::cout <<  PShapes.getIntegrationWeights()[a] << " ";
  std::cout << "\n";
  
  std::cout << "Quadrature point coordinates\n";
  for(unsigned int a=0; a<PShapes.getQuadraturePointCoordinates().size(); a++)
    std::cout <<  PShapes.getQuadraturePointCoordinates()[a] << " ";
  std::cout << "\n";
}

