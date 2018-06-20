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

/*
 * testTriangle.cpp
 * DG++
 *
 * Created by Adrian Lew on 9/7/06.
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

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Triangle.h"

int main() {
  VecDouble dummycoordinates(6);

  // Fill-in the dummy global array
  dummycoordinates[0] = 0;
  dummycoordinates[1] = 0;

  dummycoordinates[2] = 0.5;
  dummycoordinates[3] = 0.3;

  dummycoordinates[4] = 0.2;
  dummycoordinates[5] = 1.5;

  GlobalNodalIndex c[] = {0, 1, 2};
  VecSize_t conn(c, c + 3);
  Triangle<2> MyTriangle(dummycoordinates, conn);

  std::cout << "Number of vertices: " << MyTriangle.getNumVertices()
            << " should be 3\n";
  std::cout << "ParametricDimension: " << MyTriangle.getParametricDimension()
            << " should be 2\n";
  std::cout << "EmbeddingDimension: " << MyTriangle.getEmbeddingDimension()
            << " should be 2\n";

  srand(time(NULL));

  double X[2];
  X[0] = double(rand()) / double(RAND_MAX);
  X[1] = double(rand()) / double(RAND_MAX); // It may be outside the triangle

  if (MyTriangle.consistencyTest(X, 1.e-6))
    std::cout << "Consistency test successful"
              << "\n";
  else
    std::cout << "Consistency test failed"
              << "\n";

  std::cout << "\nIn Radius: " << MyTriangle.getInRadius()
            << "\nshould be 0.207002\n";
  std::cout << "\nOut Radius: " << MyTriangle.getOutRadius()
            << "\nshould be 0.790904\n";
  std::cout << "\n";

  // Faces
  ElementGeometry* face = MyTriangle.getFaceGeometry(2);
  std::cout << "Number of vertices: " << face->getNumVertices()
            << " should be 2\n";
  std::cout << "ParametricDimension: " << face->getParametricDimension()
            << " should be 1\n";
  std::cout << "EmbeddingDimension: " << face->getEmbeddingDimension()
            << " should be 2\n";
  std::cout << "Connectivity: " << face->getConnectivity()[0] << " "
            << face->getConnectivity()[1] << " should be 1 2\n";

  delete face;

  // Test virtual mechanism and copy and clone constructors
  ElementGeometry* MyElmGeo = &MyTriangle;

  std::cout << "Testing virtual mechanism: ";
  std::cout << "Polytope name: " << MyElmGeo->getPolytopeName()
            << " should be TRIANGLE\n";

  const VecSize_t& Conn = MyElmGeo->getConnectivity();
  std::cout << "Connectivity: " << Conn[0] << " " << Conn[1] << " " << Conn[2]
            << " should be 1 2 3\n";

  ElementGeometry* MyElmGeoCloned = MyTriangle.clone();
  std::cout << "Testing cloning mechanism: ";
  std::cout << "Polytope name: " << MyElmGeoCloned->getPolytopeName()
            << " should be TRIANGLE\n";
  const VecSize_t& Conn2 = MyElmGeoCloned->getConnectivity();
  std::cout << "Connectivity: " << Conn2[0] << " " << Conn2[1] << " "
            << Conn2[2] << " should be 1 2 3\n";

  std::cout << "Test triangle in 3D\n";

  VecDouble dummycoordinates3(9);

  // Fill-in the dummy global array
  dummycoordinates3[0] = 0;
  dummycoordinates3[1] = 0;
  dummycoordinates3[2] = 0;

  dummycoordinates3[3] = 0.5;
  dummycoordinates3[4] = 0.3;
  dummycoordinates3[5] = 1;

  dummycoordinates3[6] = 0.2;
  dummycoordinates3[7] = 1.5;
  dummycoordinates3[8] = 2;

  Triangle<3> MyTriangle3(dummycoordinates3, conn);

  std::cout << "Number of vertices: " << MyTriangle3.getNumVertices()
            << " should be 3\n";
  std::cout << "ParametricDimension: " << MyTriangle3.getParametricDimension()
            << " should be 2\n";
  std::cout << "EmbeddingDimension: " << MyTriangle3.getEmbeddingDimension()
            << " should be 3\n";

  srand(time(NULL));

  X[0] = double(rand()) / double(RAND_MAX);
  X[1] = double(rand()) / double(RAND_MAX); // It may be outside the triangle

  if (MyTriangle3.consistencyTest(X, 1.e-6))
    std::cout << "Consistency test successful"
              << "\n";
  else
    std::cout << "Consistency test failed"
              << "\n";

  std::cout << "\nIn Radius: " << MyTriangle3.getInRadius()
            << "\nshould be 0.26404\n";
  std::cout << "\nOut Radius: " << MyTriangle3.getOutRadius()
            << "\nshould be 1.66368\n";
  std::cout << "\n";

  return 1;
}
