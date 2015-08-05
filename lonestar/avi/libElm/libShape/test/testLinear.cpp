/*
 * testLinear.cpp
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


#include "Linear.h"
#include <iostream>

int main()
{
  /* 2D test */
  double coord2[] = {0.1,0.8};

  Linear<2> Linear2D;
  
  std::cout << Linear2D.getNumberOfFunctions() << " should be " << 3 << "\n";
  std::cout << Linear2D.getNumberOfVariables() << " should be " << 2 << "\n";

  if(Linear2D.consistencyTest(coord2,1.e-6))
    std::cout << "Consistency test successful" << "\n";
  else
    std::cout << "Consistency test failed" << "\n";
  

  std::cout << "Copy Constructor\n";
  
  Linear<2> Linear2DCopy(Linear2D);
  double flag = 0;

  for(int a=0; a<Linear2D.getNumberOfFunctions(); a++)
    if(Linear2DCopy.Val(a,coord2) != Linear2D.Val(a,coord2))
      flag = 1;
  if(flag)
      std::cout << "Copy constructor failed" << "\n";
  else
      std::cout << "Copy constructor successful" << "\n";
  
  std::cout << "Cloning and virtual mechanisms\n";

  Shape *Linear2DClone = Linear2D.clone();

  for(int a=0; a<Linear2D.getNumberOfFunctions(); a++)
    if(Linear2DClone->getVal(a,coord2) != Linear2D.Val(a,coord2))
      flag = 1;
  if(flag)
      std::cout << "Cloning failed" << "\n";
  else
      std::cout << "Cloning successful" << "\n";
  
  /* 3D test */
  std::cout << "3D test\n";

  double coord3[] = {1.2, 0.1, -0.4};

  Linear<3> Linear3D;
  
  std::cout << Linear3D.getNumberOfFunctions() << " should be " << 4 << "\n";
  std::cout << Linear3D.getNumberOfVariables() << " should be " << 3 << "\n\n";

  if(Linear3D.consistencyTest(coord3,1.e-6))
    std::cout << "Consistency test successful" << "\n";
  else
    std::cout << "Consistency test failed" << "\n";
  
}
