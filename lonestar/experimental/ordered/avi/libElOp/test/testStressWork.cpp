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

/*
 * testStressWork.cpp
 * DG++
 *
 * Created by Adrian Lew on 10/25/06.
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
#include "ElementalOperation.h"
#include "StressWork.h"
#include "P12DElement.h"

int main() {
  double Vertices[] = {1, 0, 0, 1, 0, 0, 1, 1};
  std::vector<double> Vertices0(Vertices, Vertices + 8);

  Triangle<2>::SetGlobalCoordinatesArray(Vertices0);

  P12D<2>::Bulk TestElement(1, 2, 3);
  NeoHookean NH(1, 1);

  StressWork MyResidue(&TestElement, NH, 0, 1);

  std::vector<std::vector<double>> argval(2);
  argval[0].resize(3);
  argval[1].resize(3);

  argval[0][0] = 0.1;
  argval[0][1] = 0.;
  argval[0][2] = -0.1;

  argval[1][0] = 0.;
  argval[1][1] = 0.1;
  argval[1][2] = -0.1;

  argval[0][0] = 0.;
  argval[0][1] = 0.;
  argval[0][2] = -0.;

  argval[1][0] = 0.;
  argval[1][1] = 0.;
  argval[1][2] = -0.;

  std::vector<unsigned int> DofPerField(2);

  DofPerField[0] = TestElement.getDof(MyResidue.getFields()[0]);
  DofPerField[1] = TestElement.getDof(MyResidue.getFields()[1]);

  if (MyResidue.consistencyTest(MyResidue, DofPerField, argval))
    std::cout << "DResidue::Consistency test successful\n";
  else
    std::cout << "DResidue::Consistency test not successful\n";
}
