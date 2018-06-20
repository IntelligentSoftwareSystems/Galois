/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

// Sriramajayam

// Purpose :  To check ShapesEvaluatedP13D

#include "Triangle.h"
#include "ShapesEvaluatedP22D.h"
#include <iostream>

static void PrintData(ShapesP22D::Bulk PShapes);

int main() {
  double TempVertices0[] = {0, 0, 1, 0, 0, 1};

  std::vector<double> Vertices0(TempVertices0, TempVertices0 + 6);

  Triangle<2>::SetGlobalCoordinatesArray(Vertices0);

  Triangle<2> T1(1, 2, 3);
  ShapesP22D::Bulk P1Shapes(&T1);

  std::cout << "Parametric triangle\n";
  PrintData(P1Shapes);

  std::cout << "\nTwice Parametric triangle\n";

  double TempVertices1[] = {0, 0, 2, 0, 0, 2};

  std::vector<double> Vertices1(TempVertices1, TempVertices1 + 6);
  Triangle<2>::SetGlobalCoordinatesArray(Vertices1);
  Triangle<2> T2(1, 2, 3);
  ShapesP22D::Bulk P2Shapes(&T2);

  PrintData(P2Shapes);
}

static void PrintData(ShapesP22D::Bulk PShapes) {
  std::cout << "Function values\n";
  for (unsigned int a = 0; a < PShapes.getShapes().size(); a++)
    std::cout << PShapes.getShapes()[a] << " ";
  std::cout << "\n";

  std::cout << "Function derivative values\n";
  for (unsigned int a = 0; a < PShapes.getDShapes().size(); a++)
    std::cout << PShapes.getDShapes()[a] << " ";
  std::cout << "\n";

  std::cout << "Integration weights\n";
  for (unsigned int a = 0; a < PShapes.getIntegrationWeights().size(); a++)
    std::cout << PShapes.getIntegrationWeights()[a] << " ";
  std::cout << "\n";

  std::cout << "Quadrature point coordinates\n";
  for (unsigned int a = 0; a < PShapes.getQuadraturePointCoordinates().size();
       a++)
    std::cout << PShapes.getQuadraturePointCoordinates()[a] << " ";
  std::cout << "\n";
}
