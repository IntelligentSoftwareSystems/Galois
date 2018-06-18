/*
 * testP12DElementBoundaryTraces.cpp
 * DG++
 *
 * Created by Adrian Lew on 10/12/06.
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

#include "P12DElement.h"
#include <iostream>

int main() {
  double Vertices[] = {1, 0, 0, 1, 0, 0};
  VecDouble Vertices0(Vertices, Vertices + 6);

  Triangle<2>::SetGlobalCoordinatesArray(Vertices0);
  Segment<2>::SetGlobalCoordinatesArray(Vertices0);
  ElementBoundaryTraces* TestElementBoundaryClone;

  sleep(2);

  P12DElement<2> TestElement(1, 2, 3);

  {
    P12DElementBoundaryTraces<2> TestElementBoundary(
        TestElement, true, false, true, P12DTrace<2>::TwoDofs);

    std::cout << "Number of traces: " << TestElementBoundary.getNumTraceFaces()
              << " should be 2\n";

    for (unsigned int a = 0; a < TestElementBoundary.getNumTraceFaces(); a++) {
      int facenumber = TestElementBoundary.getTraceFaceIds()[a];
      std::cout << "Face number : " << facenumber << std::endl;

      std::cout << "Normal components\n";
      for (unsigned int q = 0; q < TestElementBoundary.getNormal(a).size(); q++)
        std::cout << TestElementBoundary.getNormal(a)[q] << " ";
      std::cout << "\n";
    }

    for (unsigned int a = 0; a < TestElementBoundary.getNumTraceFaces(); a++) {
      int facenumber = TestElementBoundary.getTraceFaceIds()[a];
      std::cout << "Face number : " << facenumber << std::endl;

      std::cout << "Shape functions values for the first field\n";
      const Element& face = TestElementBoundary[a];

      for (unsigned int q = 0; q < face.getShapes(0).size(); q++)
        std::cout << face.getShapes(0)[q] << " ";

      std::cout << "\n";

      std::cout << "Integration point coordinates\n";
      for (unsigned int q = 0; q < face.getIntegrationPtCoords(0).size(); q++)
        std::cout << face.getIntegrationPtCoords(0)[q] << " ";

      std::cout << "\n";
    }

    std::cout << "\nTest copy constructor\n";

    P12DElementBoundaryTraces<2> TestElementBoundaryCopy(TestElementBoundary);

    std::cout << "Number of traces: "
              << TestElementBoundaryCopy.getNumTraceFaces() << " should be 2\n";

    for (unsigned int a = 0; a < TestElementBoundaryCopy.getNumTraceFaces();
         a++) {
      int facenumber = TestElementBoundaryCopy.getTraceFaceIds()[a];
      std::cout << "Face number : " << facenumber << std::endl;

      std::cout << "Normal components\n";
      for (unsigned int q = 0; q < TestElementBoundaryCopy.getNormal(a).size();
           q++)
        std::cout << TestElementBoundaryCopy.getNormal(a)[q] << " ";
      std::cout << "\n";
    }

    for (unsigned int a = 0; a < TestElementBoundaryCopy.getNumTraceFaces();
         a++) {
      int facenumber = TestElementBoundaryCopy.getTraceFaceIds()[a];
      std::cout << "Face number : " << facenumber << std::endl;

      std::cout << "Shape functions values for the first field\n";
      const Element& face = TestElementBoundaryCopy[a];

      for (unsigned int q = 0; q < face.getShapes(0).size(); q++)
        std::cout << face.getShapes(0)[q] << " ";

      std::cout << "\n";

      std::cout << "Integration point coordinates\n";
      for (unsigned int q = 0; q < face.getIntegrationPtCoords(0).size(); q++)
        std::cout << face.getIntegrationPtCoords(0)[q] << " ";

      std::cout << "\n";
    }

    std::cout << "\nTest Cloning\n";

    TestElementBoundaryClone = TestElementBoundary.Clone();
  }

  std::cout << "Number of traces: "
            << TestElementBoundaryClone->getNumTraceFaces() << " should be 2\n";

  for (unsigned int a = 0; a < TestElementBoundaryClone->getNumTraceFaces();
       a++) {
    int facenumber = TestElementBoundaryClone->getTraceFaceIds()[a];
    std::cout << "Face number : " << facenumber << std::endl;

    std::cout << "Normal components\n";
    for (unsigned int q = 0; q < TestElementBoundaryClone->getNormal(a).size();
         q++)
      std::cout << TestElementBoundaryClone->getNormal(a)[q] << " ";
    std::cout << "\n";
  }

  for (unsigned int a = 0; a < TestElementBoundaryClone->getNumTraceFaces();
       a++) {
    int facenumber = TestElementBoundaryClone->getTraceFaceIds()[a];
    std::cout << "Face number : " << facenumber << std::endl;

    std::cout << "Shape functions values for the first field\n";
    const Element& face = (*TestElementBoundaryClone)[a];

    for (unsigned int q = 0; q < face.getShapes(0).size(); q++)
      std::cout << face.getShapes(0)[q] << " ";

    std::cout << "\n";

    std::cout << "Integration point coordinates\n";
    for (unsigned int q = 0; q < face.getIntegrationPtCoords(0).size(); q++)
      std::cout << face.getIntegrationPtCoords(0)[q] << " ";

    std::cout << "\n";
  }
  delete TestElementBoundaryClone;

  {
    std::cout << "\n Test ThreeDofs traces\n";

    P12DElementBoundaryTraces<2> TestElementBoundary(
        TestElement, true, false, true, P12DTrace<2>::ThreeDofs);

    std::cout << "Number of traces: " << TestElementBoundary.getNumTraceFaces()
              << " should be 2\n";

    for (unsigned int a = 0; a < TestElementBoundary.getNumTraceFaces(); a++) {
      int facenumber = TestElementBoundary.getTraceFaceIds()[a];
      std::cout << "Face number : " << facenumber << std::endl;

      std::cout << "Normal components\n";
      for (unsigned int q = 0; q < TestElementBoundary.getNormal(a).size(); q++)
        std::cout << TestElementBoundary.getNormal(a)[q] << " ";
      std::cout << "\n";
    }

    for (unsigned int a = 0; a < TestElementBoundary.getNumTraceFaces(); a++) {
      int facenumber = TestElementBoundary.getTraceFaceIds()[a];
      std::cout << "Face number : " << facenumber << std::endl;

      std::cout << "Shape functions values for the first field\n";
      const Element& face = TestElementBoundary[a];

      for (unsigned int q = 0; q < face.getShapes(0).size(); q++)
        std::cout << face.getShapes(0)[q] << " ";

      std::cout << "\n";

      std::cout << "Integration point coordinates\n";
      for (unsigned int q = 0; q < face.getIntegrationPtCoords(0).size(); q++)
        std::cout << face.getIntegrationPtCoords(0)[q] << " ";

      std::cout << "\n";
    }
  }
}
