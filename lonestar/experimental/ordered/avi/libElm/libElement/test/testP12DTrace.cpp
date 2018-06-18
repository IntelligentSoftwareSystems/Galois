/*
 * testP12DElement.cpp
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

  sleep(2);

  P12DElement<2> TestElement(1, 2, 3);

  P12DTrace<2> TestTraceOne(TestElement, P12DTrace<2>::FaceOne,
                            P12DTrace<2>::TwoDofs);
  P12DTrace<2> TestTraceTwo(TestElement, P12DTrace<2>::FaceTwo,
                            P12DTrace<2>::TwoDofs);
  P12DTrace<2> TestTraceThree(TestElement, P12DTrace<2>::FaceThree,
                              P12DTrace<2>::TwoDofs);

  P12DTrace<2>* Faces[] = {&TestTraceOne, &TestTraceTwo, &TestTraceThree};

  for (int i = 0; i < 3; i++) {
    std::cout << "\nTesting Face: " << i + 1 << "\n";
    std::cout << "Number of fields: " << Faces[i]->getFields()
              << " should be 2\n";
    std::cout << "Number of dof field(0): " << Faces[i]->getDof(0)
              << " should be 2\n";
    std::cout << "Number of dof field(1): " << Faces[i]->getDof(1)
              << " should be 2\n";

    for (int a = 0; a < 2; a++) {
      std::cout << "Shape function values at quad points field(" << a << "):\n";
      for (unsigned int q = 0; q < Faces[i]->getShapes(a).size(); q++)
        std::cout << Faces[i]->getShapes(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Integration weight values at quad points field(" << a
                << "):\n";
      for (unsigned int q = 0; q < Faces[i]->getIntegrationWeights(a).size();
           q++)
        std::cout << Faces[i]->getIntegrationWeights(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Quad points coordinates for field(" << a << "):\n";
      for (unsigned int q = 0; q < Faces[i]->getIntegrationPtCoords(a).size();
           q++)
        std::cout << Faces[i]->getIntegrationPtCoords(a)[q] << " ";
      std::cout << "\n";
    }
    std::cout << "No shape function derivatives tested\n";
  }

  P12DTrace<2>* VirtualTraceOne;

  {
    P12DTrace<2> CopyTraceOne(TestTraceOne);
    std::cout << "\nTest Copy Constructor for Face 1\n";

    std::cout << "Number of fields: " << CopyTraceOne.getFields()
              << " should be 2\n";
    std::cout << "Number of dof field(0): " << CopyTraceOne.getDof(0)
              << " should be 2\n";
    std::cout << "Number of dof field(1): " << CopyTraceOne.getDof(1)
              << " should be 2\n";

    for (int a = 0; a < 2; a++) {
      std::cout << "Shape function values at quad points field(" << a << "):\n";
      for (unsigned int q = 0; q < CopyTraceOne.getShapes(a).size(); q++)
        std::cout << CopyTraceOne.getShapes(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Integration weight values at quad points field(" << a
                << "):\n";
      for (unsigned int q = 0; q < CopyTraceOne.getIntegrationWeights(a).size();
           q++)
        std::cout << CopyTraceOne.getIntegrationWeights(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Quad points coordinates for field(" << a << "):\n";
      for (unsigned int q = 0;
           q < CopyTraceOne.getIntegrationPtCoords(a).size(); q++)
        std::cout << CopyTraceOne.getIntegrationPtCoords(a)[q] << " ";
      std::cout << "\n";
    }
    std::cout << "No shape function derivatives tested\n";

    VirtualTraceOne = CopyTraceOne.clone();
    std::cout
        << "\nCloned element before destruction. Test cloning mechanism\n";
  }

  std::cout << "Number of fields: " << VirtualTraceOne->getFields()
            << " should be 2\n";
  std::cout << "Number of dof field(0): " << VirtualTraceOne->getDof(0)
            << " should be 2\n";
  std::cout << "Number of dof field(1): " << VirtualTraceOne->getDof(1)
            << " should be 2\n";

  for (int a = 0; a < 2; a++) {
    std::cout << "Shape function values at quad points field(" << a << "):\n";
    for (unsigned int q = 0; q < VirtualTraceOne->getShapes(a).size(); q++)
      std::cout << VirtualTraceOne->getShapes(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Integration weight values at quad points field(" << a
              << "):\n";
    for (unsigned int q = 0;
         q < VirtualTraceOne->getIntegrationWeights(a).size(); q++)
      std::cout << VirtualTraceOne->getIntegrationWeights(a)[q] << " ";
    std::cout << "\n";
  }

  for (int a = 0; a < 2; a++) {
    std::cout << "Quad points coordinates for field(" << a << "):\n";
    for (unsigned int q = 0;
         q < VirtualTraceOne->getIntegrationPtCoords(a).size(); q++)
      std::cout << VirtualTraceOne->getIntegrationPtCoords(a)[q] << " ";
    std::cout << "\n";
  }
  std::cout << "No shape function derivatives tested\n\n";

  delete VirtualTraceOne;

  std::cout << "Test different ShapeType\n";
  P12DTrace<2> TestTraceOneType(TestElement, P12DTrace<2>::FaceOne,
                                P12DTrace<2>::ThreeDofs);
  P12DTrace<2> TestTraceTwoType(TestElement, P12DTrace<2>::FaceTwo,
                                P12DTrace<2>::ThreeDofs);
  P12DTrace<2> TestTraceThreeType(TestElement, P12DTrace<2>::FaceThree,
                                  P12DTrace<2>::ThreeDofs);

  Faces[0] = &TestTraceOneType;
  Faces[1] = &TestTraceTwoType;
  Faces[2] = &TestTraceThreeType;

  for (int i = 0; i < 3; i++) {
    std::cout << "\nTesting Face: " << i + 1 << "\n";
    std::cout << "Number of fields: " << Faces[i]->getFields()
              << " should be 2\n";
    std::cout << "Number of dof field(0): " << Faces[i]->getDof(0)
              << " should be 3\n";
    std::cout << "Number of dof field(1): " << Faces[i]->getDof(1)
              << " should be 3\n";

    for (int a = 0; a < 2; a++) {
      std::cout << "Shape function values at quad points field(" << a << "):\n";
      for (unsigned int q = 0; q < Faces[i]->getShapes(a).size(); q++)
        std::cout << Faces[i]->getShapes(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Integration weight values at quad points field(" << a
                << "):\n";
      for (unsigned int q = 0; q < Faces[i]->getIntegrationWeights(a).size();
           q++)
        std::cout << Faces[i]->getIntegrationWeights(a)[q] << " ";
      std::cout << "\n";
    }

    for (int a = 0; a < 2; a++) {
      std::cout << "Quad points coordinates for field(" << a << "):\n";
      for (unsigned int q = 0; q < Faces[i]->getIntegrationPtCoords(a).size();
           q++)
        std::cout << Faces[i]->getIntegrationPtCoords(a)[q] << " ";
      std::cout << "\n";
    }
    std::cout << "No shape function derivatives tested\n";
  }
}
