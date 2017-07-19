/*
 * testSegment.cpp
 * DG++
 *
 * Created by Adrian Lew on 10/8/06.
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

#include "Segment.h"


int main()
{
  VecDouble dummycoordinates(4);

  // Fill-in the dummy global array
  dummycoordinates[0] = 0;
  dummycoordinates[1] = 0;

  dummycoordinates[2] = 0.5;
  dummycoordinates[3] = 0.3;

 
  VecSize_t conn(2);
  conn[0] = 0;
  conn[1] = 1;
  Segment<2> MySegment(dummycoordinates, conn);

  std::cout << "Number of vertices: " << MySegment.getNumVertices() << " should be 2\n";
  std::cout << "ParametricDimension: " << MySegment.getParametricDimension() << " should be 1\n";
  std::cout << "EmbeddingDimension: " << MySegment.getEmbeddingDimension() << " should be 2\n";
  
  srand(time(NULL));

  double X[2];
  X[0] = double(rand())/double(RAND_MAX);  //It may be outside the segment

  if(MySegment.consistencyTest(X,1.e-6))
    std::cout << "Consistency test successful" << "\n";
  else
    std::cout << "Consistency test failed" << "\n";

  // Test virtual mechanism and copy and clone constructors
  ElementGeometry *MyElmGeo = &MySegment;
  
  std::cout << "Testing virtual mechanism: ";
  std::cout << "Polytope name: " << MyElmGeo->getPolytopeName() << " should be SEGMENT\n";
  
  const  VecSize_t  &Conn = MyElmGeo->getConnectivity();
  std::cout << "Connectivity: " << Conn[0] << " " << Conn[1]  << " should be 1 2\n"; 

  ElementGeometry *MyElmGeoCloned = MySegment.clone();
  std::cout << "Testing cloning mechanism: ";
  std::cout << "Polytope name: " << MyElmGeoCloned->getPolytopeName() << " should be SEGMENT\n";
  const VecSize_t  &Conn2 = MyElmGeoCloned->getConnectivity();
  std::cout << "Connectivity: " << Conn2[0] << " " << Conn2[1]  << " should be 1 2\n"; 
  


  std::cout << "Test Segment in 3D\n";

  VecDouble dummycoordinates3(6);

  // Fill-in the dummy global array
  dummycoordinates3[0] = 0;
  dummycoordinates3[1] = 0;
  dummycoordinates3[2] = 0;

  dummycoordinates3[3] = 0.5;
  dummycoordinates3[4] = 0.3;
  dummycoordinates3[5] = 1;
  

  Segment<3> MySegment3(dummycoordinates3, conn);

  std::cout << "Number of vertices: " << MySegment3.getNumVertices() << " should be 2\n";
  std::cout << "ParametricDimension: " << MySegment3.getParametricDimension() << " should be 1\n";
  std::cout << "EmbeddingDimension: " << MySegment3.getEmbeddingDimension() << " should be 3\n";
  
  srand(time(NULL));

  X[0] = double(rand())/double(RAND_MAX); // It may be outside the segment

  if(MySegment3.consistencyTest(X,1.e-6))
    std::cout << "Consistency test successful" << "\n";
  else
    std::cout << "Consistency test failed" << "\n";



  return 1;

}
