/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

// Purpose: To check P13DElement.

#include "P13DElement.h"
#include <iostream>
#include <vector>

int main() {
  double Vertices[] = {1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
  VecDouble Vertices0(Vertices, Vertices + 12);

  Tetrahedron::SetGlobalCoordinatesArray(Vertices0);

  P13DElement<2> TestElement(1, 2, 3, 4);

  Element* VirtualElement;

  sleep(2);

  std::cout << "Number of fields: " << TestElement.getFields()
            << " should be 2\n";
  std::cout << "Number of dof field(0): " << TestElement.getDof(0)
            << " should be 4\n";
  std::cout << "Number of dof field(1): " << TestElement.getDof(1)
            << " should be 4\n";
  for (int a = 0; a < 2; a++) {
    std::cout << "Shape function values at quad points field(" << a << "):\n";
    for (unsigned int q = 0; q < TestElement.getShapes(a).size(); q++)
      std::cout << TestElement.getShapes(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Shape function derivatives values at quad points field(" << a
              << "):\n";
    for (unsigned int q = 0; q < TestElement.getDShapes(a).size(); q++)
      std::cout << TestElement.getDShapes(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Integration weight values at quad points field(" << a
              << "):\n";
    for (unsigned int q = 0; q < TestElement.getIntegrationWeights(a).size();
         q++)
      std::cout << TestElement.getIntegrationWeights(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Quad points coordinates for field(" << a << "):\n";
    for (unsigned int q = 0; q < TestElement.getIntegrationPtCoords(a).size();
         q++)
      std::cout << TestElement.getIntegrationPtCoords(a)[q] << " ";
    std::cout << "\n";
  }

  // testing CopyConstructor : //
  {
    P13DElement<2> CopyElement(TestElement);
    std::cout << "Test Copy Constructor\n";

    std::cout << "Number of fields: " << CopyElement.getFields()
              << " should be 2\n";
    std::cout << "Number of dof field(0): " << CopyElement.getDof(0)
              << " should be 4\n";
    std::cout << "Number of dof field(1): " << CopyElement.getDof(1)
              << " should be 4\n";

    for (int a = 0; a < 2; a++) {
      std::cout << "Shape function values at quad points field(" << a << "):\n";
      for (unsigned int q = 0; q < CopyElement.getShapes(a).size(); q++)
        std::cout << CopyElement.getShapes(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Shape function derivatives values at quad points field("
                << a << "):\n";
      for (unsigned int q = 0; q < CopyElement.getDShapes(a).size(); q++)
        std::cout << CopyElement.getDShapes(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Integration weight values at quad points field(" << a
                << "):\n";
      for (unsigned int q = 0; q < CopyElement.getIntegrationWeights(a).size();
           q++)
        std::cout << CopyElement.getIntegrationWeights(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Quad points cooridnates for field(" << a << "):\n";
      for (unsigned int q = 0; q < CopyElement.getIntegrationPtCoords(a).size();
           q++)
        std::cout << CopyElement.getIntegrationPtCoords(a)[q] << " ";
      std::cout << "\n";
    }

    // Cloning mechanism : //
    VirtualElement = CopyElement.clone();
    std::cout << "Cloned element before destruction. Test cloning mechanism\n";
  }

  std::cout << "Number of fields: " << VirtualElement->getNumFields()
            << " should be 2\n";
  std::cout << "Number of dof field(0): " << VirtualElement->getDof(0)
            << " should be 4\n";
  std::cout << "Number of dof field(1): " << VirtualElement->getDof(1)
            << " should be 4\n";

  for (int a = 0; a < 2; a++) {
    std::cout << "Shape function values at quad points field(" << a << "):\n";
    for (unsigned int q = 0; q < VirtualElement->getShapes(a).size(); q++)
      std::cout << VirtualElement->getShapes(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Shape function derivatives values at quad points field(" << a
              << "):\n";
    for (unsigned int q = 0; q < VirtualElement->getDShapes(a).size(); q++)
      std::cout << VirtualElement->getDShapes(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Integration weight values at quad points field(" << a
              << "):\n";
    for (unsigned int q = 0;
         q < VirtualElement->getIntegrationWeights(a).size(); q++)
      std::cout << VirtualElement->getIntegrationWeights(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Quad points cooridnates for field(" << a << "):\n";
    for (unsigned int q = 0;
         q < VirtualElement->getIntegrationPtCoords(a).size(); q++)
      std::cout << VirtualElement->getIntegrationPtCoords(a)[q] << " ";
    std::cout << "\n";
  }

  delete VirtualElement;
}
