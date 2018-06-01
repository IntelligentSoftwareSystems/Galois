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
#include <iostream>

static void PrintData(ShapesP12D::Bulk P10Shapes);

int main()
{
  double V0[] = {1,0,0,1,0,0};
  std::vector<double> Vertices0(V0, V0+6);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices0);
  Triangle<2> P10(1,2,3);
  ShapesP12D::Bulk P10Shapes(&P10);
  
  std::cout << "Parametric triangle\n";
  PrintData(P10Shapes);

  std::cout << "\nTwice Parametric triangle\n";

  double V1[] = {2,0,0,2,0,0};
  std::vector<double> Vertices1(V1, V1+6);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices1);
  Triangle<2> P11(1,2,3);
  ShapesP12D::Bulk P11Shapes(&P11);

  PrintData(P11Shapes);

  std::cout << "\nReordered nodes of  twice parametric triangle\n";

  double V2[] = {0,0,2,0,0,2};
  std::vector<double> Vertices2(V2, V2+6);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices2);
  Triangle<2> P12(1,2,3);
  ShapesP12D::Bulk P12Shapes(&P12);

  PrintData(P12Shapes);

  std::cout << "\n Equilateral triangle with area sqrt(3)\n";

  double V3[] = {0,0,2,0,1,sqrt(3)};
  std::vector<double> Vertices3(V3, V3+6);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices3);
  Triangle<2> P13(1,2,3);
  ShapesP12D::Bulk P13Shapes(&P13);

  PrintData(P13Shapes);

  std::cout << "\n Irregular triangle with area sqrt(3)\n";

  double V4[] = {0,0,2,0,2.5,sqrt(3)};
  std::vector<double> Vertices4(V4, V4+6);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices4);
  Triangle<2> P14(1,2,3);
  ShapesP12D::Bulk P14Shapes(&P14);

  PrintData(P14Shapes);

}




static void PrintData(ShapesP12D::Bulk P10Shapes)
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
