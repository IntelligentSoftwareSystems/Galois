// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GEOMETRY_DATA
#define _GEOMETRY_DATA

#include "geometry.h"
#include "dataGen.h"

point2d rand2d(unsigned int i) {
  unsigned int s1 = i;
  unsigned int s2 = i + dataGen::hash<unsigned int>(s1);
  return point2d(2*dataGen::hash<double>(s1)-1,
		 2*dataGen::hash<double>(s2)-1);
}

point3d rand3d(int i) {
  unsigned int s1 = i;
  unsigned int s2 = i + dataGen::hash<unsigned int>(s1);
  unsigned int s3 = 2*i + dataGen::hash<unsigned int>(s2);
  return point3d(2*dataGen::hash<double>(s1)-1,
		 2*dataGen::hash<double>(s2)-1,
		 2*dataGen::hash<double>(s3)-1);
}

point2d randInUnitSphere2d(int i) {
  int j = 0;
  vect2d v;
  do {
    int o = dataGen::hash<int>(j++);
    v = vect2d(rand2d(o+i));
  } while (v.Length() > 1.0);
  return point2d(v);
}

point3d randInUnitSphere3d(int i) {
  int j = 0;
  vect3d v;
  do {
    int o = dataGen::hash<int>(j++);
    v = vect3d(rand3d(o+i));
  } while (v.Length() > 1.0);
  return point3d(v);
}

point2d randOnUnitSphere2d(int i) {
  vect2d v = vect2d(randInUnitSphere2d(i));
  return point2d(v/v.Length());
}

point3d randOnUnitSphere3d(int i) {
  vect3d v = vect3d(randInUnitSphere3d(i));
  return point3d(v/v.Length());
}

#endif // _GEOMETRY_DATA
