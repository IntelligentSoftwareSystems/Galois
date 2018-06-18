/*
 * testSimpleQuadrature.cpp
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

#include "Quadrature.h"
#include <iostream>

int main() {
  std::cout << Triangle_1::Bulk->getNumQuadraturePoints() << " should be " << 3
            << "\n";
  std::cout << Triangle_1::Bulk->getNumShapeCoordinates() << " should be " << 2
            << "\n\n";
  std::cout << Triangle_1::Bulk->getNumCoordinates() << " should be " << 2
            << "\n\n";

  for (int q = 0; q < Triangle_1::Bulk->getNumQuadraturePoints(); q++) {
    for (int i = 0; i < Triangle_1::Bulk->getNumCoordinates(); i++)
      std::cout << Triangle_1::Bulk->getQuadraturePoint(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.666667 0.166667 \n"
               "0.166667 0.666667 \n"
               "0.166667 0.166667 \n\n";

  for (int q = 0; q < Triangle_1::Bulk->getNumQuadraturePoints(); q++) {
    for (int i = 0; i < Triangle_1::Bulk->getNumShapeCoordinates(); i++)
      std::cout << Triangle_1::Bulk->getQuadraturePointShape(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.666667 0.166667 \n"
               "0.166667 0.666667 \n"
               "0.166667 0.166667 \n\n";

  for (int q = 0; q < Triangle_1::Bulk->getNumQuadraturePoints(); q++)
    std::cout << Triangle_1::Bulk->getQuadratureWeights(q) << " ";
  std::cout << "\n";

  std::cout << "should read \n"
               "0.166667 0.166667 0.166667\n\n";

  std::cout << "\n Copy Constructor\n";

  Quadrature GaussCopy(*Triangle_1::Bulk);
  std::cout << GaussCopy.getNumQuadraturePoints() << " should be " << 3 << "\n";
  std::cout << GaussCopy.getNumShapeCoordinates() << " should be " << 2
            << "\n\n";
  std::cout << GaussCopy.getNumCoordinates() << " should be " << 2 << "\n\n";
  for (int q = 0; q < GaussCopy.getNumQuadraturePoints(); q++) {
    for (int i = 0; i < GaussCopy.getNumCoordinates(); i++)
      std::cout << GaussCopy.getQuadraturePoint(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.666667 0.166667 \n"
               "0.166667 0.666667 \n"
               "0.166667 0.166667 \n\n";

  for (int q = 0; q < GaussCopy.getNumQuadraturePoints(); q++) {
    for (int i = 0; i < GaussCopy.getNumShapeCoordinates(); i++)
      std::cout << GaussCopy.getQuadraturePointShape(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.666667 0.166667 \n"
               "0.166667 0.666667 \n"
               "0.166667 0.166667 \n\n";

  for (int q = 0; q < GaussCopy.getNumQuadraturePoints(); q++)
    std::cout << GaussCopy.getQuadratureWeights(q) << " ";
  std::cout << "\n";

  std::cout << "should read \n "
               "0.166667 0.166667 0.166667\n\n";

  std::cout << "\n Cloning and virtual mechanisms\n";

  Quadrature* GaussClone = Triangle_1::Bulk->clone();
  std::cout << GaussClone->getNumQuadraturePoints() << " should be " << 3
            << "\n";
  std::cout << GaussClone->getNumCoordinates() << " should be " << 2 << "\n\n";
  for (int q = 0; q < GaussClone->getNumQuadraturePoints(); q++) {
    for (int i = 0; i < GaussClone->getNumCoordinates(); i++)
      std::cout << GaussClone->getQuadraturePoint(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.666667 0.166667 \n"
               "0.166667 0.666667 \n"
               "0.166667 0.166667 \n\n";

  for (int q = 0; q < GaussClone->getNumQuadraturePoints(); q++)
    std::cout << GaussClone->getQuadratureWeights(q) << " ";
  std::cout << "\n";

  std::cout << "should read \n "
               "0.166667 0.166667 0.166667\n\n";

  std::cout << Triangle_1::FaceOne->getNumQuadraturePoints() << " should be "
            << 2 << "\n";
  std::cout << Triangle_1::FaceOne->getNumCoordinates() << " should be " << 1
            << "\n\n";
  std::cout << Triangle_1::FaceOne->getNumShapeCoordinates() << " should be "
            << 2 << "\n\n";

  for (int q = 0; q < Triangle_1::FaceOne->getNumQuadraturePoints(); q++) {
    for (int i = 0; i < Triangle_1::FaceOne->getNumCoordinates(); i++)
      std::cout << Triangle_1::FaceOne->getQuadraturePoint(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.788675\n"
               "0.211325\n\n";

  for (int q = 0; q < Triangle_1::FaceOne->getNumQuadraturePoints(); q++) {
    for (int i = 0; i < Triangle_1::FaceOne->getNumShapeCoordinates(); i++)
      std::cout << Triangle_1::FaceOne->getQuadraturePointShape(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.788675 0.211325\n"
               "0.211325 0.788675\n\n";

  for (int q = 0; q < Triangle_1::FaceOne->getNumQuadraturePoints(); q++)
    std::cout << Triangle_1::FaceOne->getQuadratureWeights(q) << " ";
  std::cout << "\n";

  std::cout << "should read \n"
               "0.5 0.5\n\n";

  std::cout << "Test copy constructor once more\n\n";

  Quadrature NewTriangleFace(*Triangle_1::FaceOne);

  std::cout << NewTriangleFace.getNumQuadraturePoints() << " should be " << 2
            << "\n";
  std::cout << NewTriangleFace.getNumCoordinates() << " should be " << 1
            << "\n\n";
  std::cout << NewTriangleFace.getNumShapeCoordinates() << " should be " << 2
            << "\n\n";

  for (int q = 0; q < NewTriangleFace.getNumQuadraturePoints(); q++) {
    for (int i = 0; i < NewTriangleFace.getNumCoordinates(); i++)
      std::cout << NewTriangleFace.getQuadraturePoint(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.788675\n"
               "0.211325\n\n";

  for (int q = 0; q < NewTriangleFace.getNumQuadraturePoints(); q++) {
    for (int i = 0; i < NewTriangleFace.getNumShapeCoordinates(); i++)
      std::cout << NewTriangleFace.getQuadraturePointShape(q)[i] << " ";
    std::cout << "\n";
  }
  std::cout << "should read \n"
               "0.788675 0.211325\n"
               "0.211325 0.788675\n\n";

  for (int q = 0; q < NewTriangleFace.getNumQuadraturePoints(); q++)
    std::cout << NewTriangleFace.getQuadratureWeights(q) << " ";
  std::cout << "\n";

  std::cout << "should read \n"
               "0.5 0.5\n\n";
}
