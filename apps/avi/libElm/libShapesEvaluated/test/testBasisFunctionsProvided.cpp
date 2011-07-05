/*
 * testLinearSE.cpp
 * DG++
 *
 * Created by Adrian Lew on 9/9/06.
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

#include "Triangle.h"
#include "ShapesEvaluated.h"
#include "BasisFunctionsProvided.h"
#include <iostream>

static void PrintData(const BasisFunctions & P10Shapes);

int main()
{
  double V0[] = {1,0,0,1,0,0};
  std::vector<double> Vertices0(V0, V0+6);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices0);
  Triangle<2> P10(1,2,3);
  ShapesP12D::Bulk * P10Shapes = new ShapesP12D::Bulk(&P10);

  
  BasisFunctionsProvided  BFS(P10Shapes->getShapes(), P10Shapes->getDShapes(), 
			      P10Shapes->getIntegrationWeights(), P10Shapes->getQuadraturePointCoordinates());

  BasisFunctionsProvidedExternalQuad  BFSExternalQuad(P10Shapes->getShapes(), P10Shapes->getDShapes(), 
						      P10Shapes->getIntegrationWeights(), 
						      P10Shapes->getQuadraturePointCoordinates());

  BasisFunctionsProvidedExternalQuad
    BFSExternalQuadNoShapes(2,
			    P10Shapes->getDShapes(), 
			    P10Shapes->getIntegrationWeights(), 
			    P10Shapes->getQuadraturePointCoordinates());
  
  std::cout << "\nPrint data before deleting original shapes\n";
  std::cout << "\n\nTest BasisFunctionProvided\n";
  PrintData(BFS);

  std::cout << "\n\nTest BasisFunctionProvidedExternalQuad\n";
  PrintData(BFSExternalQuad);

  std::cout << "\n\nTest BasisFunctionProvidedExternalQuadNoShapes\n";
  PrintData(BFSExternalQuadNoShapes);

  std::cout << "\n\nTest Copy constructors\n";
  BasisFunctionsProvided BFSCopy(BFS);
  BasisFunctionsProvidedExternalQuad BFSExternalQuadCopy(BFSExternalQuad);

  std::cout << "\n\nTest BasisFunctionProvided\n";
  PrintData(BFSCopy);

  std::cout << "\n\nTest BasisFunctionProvidedExternalQuad\n";
  PrintData(BFSExternalQuadCopy);
  

  std::cout << "\n\nTest cloning\n";
  BasisFunctions * BFSClone = BFS.clone();
  BasisFunctions * BFSExternalQuadClone = BFSExternalQuad.clone();

  std::cout << "\n\nTest BasisFunctionProvided\n";
  PrintData(*BFSClone);

  std::cout << "\n\nTest BasisFunctionProvidedExternalQuad\n";
  PrintData(*BFSExternalQuadClone);
      
  delete BFSClone;
  delete BFSExternalQuadClone;
  
  delete P10Shapes;

  std::cout << "\nPrint data  after deleted original shapes \n";
  PrintData(BFS);
  

}



static void PrintData(const BasisFunctions & P10Shapes)
{
  std::cout << "Function values\n";
  for(unsigned int a=0; a<P10Shapes.getShapes().size(); a++)
    std::cout <<  P10Shapes.getShapes()[a] << " ";
  std::cout << "\n";
  
  std::cout << "Function derivative values\n";
  for(unsigned int a=0; a<P10Shapes.getDShapes().size(); a++)
    std::cout <<  P10Shapes.getDShapes()[a] << " ";
  std::cout << "\n";
  
  std::cout << "Integration weights\n";
  for(unsigned int a=0; a<P10Shapes.getIntegrationWeights().size(); a++)
    std::cout <<  P10Shapes.getIntegrationWeights()[a] << " ";
  std::cout << "\n";
  
  std::cout << "Quadrature point coordinates\n";
  for(unsigned int a=0; a<P10Shapes.getQuadraturePointCoordinates().size(); a++)
    std::cout <<  P10Shapes.getQuadraturePointCoordinates()[a] << " ";
  std::cout << "\n";
}
