/**
 * Triangle.h
 * DG++
 *
 * Created by Adrian Lew on 9/4/06.
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

#ifndef TRIANGLE
#define TRIANGLE

#include "AuxDefs.h"
#include "ElementGeometry.h"
#include "Segment.h"

#include <algorithm>

#include <cmath>
#include <cassert>



/**
   \brief Triangle: Geometry of planar triangles 
   
   A Triangle is:\n
   1) A set of indices that describe the connectivity of the triangle, 
   properly oriented. The coordinates
   are not stored in the element but wherever the application decides\n
   2) An affine map from a two-dimensional triangle (parametric configuration) 
   with area 1/2 to the convex  
   hull of the three vertices. Triangles embedded in three-dimensional space 
   are hence easily handled. \n
   
   The parametric configuration is the triangle (0,0),(1,0),(0,1).\n
   The two parametric coordinates used are the ones aligned with the two axes in 2D.

   For a triangle with connectivity (1,2,3) the faces are ordered as
   (1,2),(2,3),(3,1).

   Rationale for a templated class: prevent having multiple copies of
   the dimension of SPD in each one of the multiple
   elements in a mesh. A static variable for it would have only
   allowed to use one type of triangles in a program.

   \warning Neither map nor dMap check for bounds of
   their array arguments
*/


template<size_t SPD> 
class Triangle: public AbstractGeom<SPD> {
 public:
  //! Connectivity in Triangle<SPD> GlobalCoordinatesArray
  Triangle (const VecDouble& globalCoordVec, const VecSize_t& connectivity)
    :AbstractGeom<SPD> (globalCoordVec, connectivity) {
      assert (connectivity.size () == 3);
  }


  inline virtual ~Triangle(){}

  Triangle(const Triangle<SPD> & that) : AbstractGeom<SPD>(that) {
  }

  virtual Triangle<SPD>* clone() const {
    return new Triangle<SPD>(*this);
  }


  inline size_t getNumVertices() const { return 3; }

  inline const std::string getPolytopeName() const { return "TRIANGLE"; }

  inline size_t getParametricDimension() const { return 2; }

  inline size_t getEmbeddingDimension() const { return SPD; }

  void map(const double * X, double *Y) const; 

  void dMap(const double * X, double *DY, double &Jac) const; 

  inline size_t getNumFaces() const { return 3; }

  virtual double getInRadius(void) const;

  virtual double getOutRadius(void) const;

  virtual Segment<SPD> * getFaceGeometry(size_t e) const; 

  virtual void computeNormal (size_t e, VecDouble& vNormal) const;

  virtual void computeCenter (VecDouble& center) const;

private:
  static size_t    SegmentNodes[];

  static double ParamCoord[];

  static double midpoint (double x1, double x2) { return (x1 + x2) / 2; }
};


// Class implementation

template <size_t SPD>
size_t    Triangle<SPD>::SegmentNodes[] = {0,1,1,2,2,0};

template <size_t SPD>
double Triangle<SPD>::ParamCoord[] = {1,0,0,1,0,0};


template<size_t SPD> 
void Triangle<SPD>::map(const double * X, double *Y) const
{
  for(size_t i=0; i<SPD; i++)
    Y[i] = X[0]*AbstractGeom<SPD>::getCoordinate(0,i) + X[1]*AbstractGeom<SPD>::getCoordinate(1,i) + (1-X[0]-X[1])*AbstractGeom<SPD>::getCoordinate(2,i);

  return;
}




template<size_t SPD> 
void Triangle<SPD>::dMap(const double * X, double *DY, double &Jac) const
{
  for(size_t i=0; i<SPD; i++)
    {
      DY[                  i] = AbstractGeom<SPD>::getCoordinate(0,i) - AbstractGeom<SPD>::getCoordinate(2,i);
      DY[SPD+i] = AbstractGeom<SPD>::getCoordinate(1,i) - AbstractGeom<SPD>::getCoordinate(2,i);
    }
  
  double g11=0;
  double g22=0;
  double g12=0;

  for(size_t i=0; i<SPD; i++)
    {
      g11 += DY[i]*DY[i];
      g22 += DY[SPD+i]*DY[SPD+i];
      g12 += DY[SPD+i]*DY[i];
    }
  
  Jac=sqrt(g11*g22-g12*g12);
  
  return;
}


template<size_t SPD>   
Segment<SPD> * Triangle<SPD>::getFaceGeometry(size_t e) const
{
  VecSize_t conn(2);
  switch(e)
  {
    case 0:
      conn[0] = AbstractGeom<SPD>::getConnectivity ()[0];
      conn[1] = AbstractGeom<SPD>::getConnectivity ()[1];
      break;

    case 1:
      conn[0] = AbstractGeom<SPD>::getConnectivity ()[1];
      conn[1] = AbstractGeom<SPD>::getConnectivity ()[2];
      break;

    case 2:
      conn[0] = AbstractGeom<SPD>::getConnectivity ()[2];
      conn[1] = AbstractGeom<SPD>::getConnectivity ()[0];
      break;

    default:
      return 0;
  }

  return new Segment<SPD> (AbstractGeom<SPD>::getGlobalCoordVec (), conn);
}

template<size_t SPD> 
double Triangle<SPD>:: getInRadius(void) const {
  double a,b,c,s;
  a = b = c = s = 0.0;
  for(size_t i=0; i<SPD; i++) {
    a += (AbstractGeom<SPD>::getCoordinate(1,i) - AbstractGeom<SPD>::getCoordinate(0,i))*
      (AbstractGeom<SPD>::getCoordinate(1,i) - AbstractGeom<SPD>::getCoordinate(0,i)) ;
    b += (AbstractGeom<SPD>::getCoordinate(2,i) - AbstractGeom<SPD>::getCoordinate(1,i))*
      (AbstractGeom<SPD>::getCoordinate(2,i) - AbstractGeom<SPD>::getCoordinate(1,i)) ;
    c += (AbstractGeom<SPD>::getCoordinate(0,i) - AbstractGeom<SPD>::getCoordinate(2,i))*
      (AbstractGeom<SPD>::getCoordinate(0,i) - AbstractGeom<SPD>::getCoordinate(2,i)) ;
  }
  a = sqrt(a);
  b = sqrt(b);
  c = sqrt(c);
  s = (a + b + c)/2.0;
  return(2.0*sqrt(s*(s-a)*(s-b)*(s-c))/(a+b+c));
}


template<size_t SPD> 
double Triangle<SPD>:: getOutRadius(void) const {
  double a,b,c;
  a = b = c = 0.0;
  for(size_t i=0; i<SPD; i++) {
    a += (AbstractGeom<SPD>::getCoordinate(1,i) - AbstractGeom<SPD>::getCoordinate(0,i))*
      (AbstractGeom<SPD>::getCoordinate(1,i) - AbstractGeom<SPD>::getCoordinate(0,i)) ;
    b += (AbstractGeom<SPD>::getCoordinate(2,i) - AbstractGeom<SPD>::getCoordinate(1,i))*
      (AbstractGeom<SPD>::getCoordinate(2,i) - AbstractGeom<SPD>::getCoordinate(1,i)) ;
    c += (AbstractGeom<SPD>::getCoordinate(0,i) - AbstractGeom<SPD>::getCoordinate(2,i))*
      (AbstractGeom<SPD>::getCoordinate(0,i) - AbstractGeom<SPD>::getCoordinate(2,i)) ;
  }
  a = sqrt(a);
  b = sqrt(b);
  c = sqrt(c);
  return(a*b*c/sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c)));
}

template <size_t SPD>
void Triangle<SPD>::computeNormal(size_t e, VecDouble& VNormal) const {
  double NodalCoord[4];

  size_t    n[2];
  double v[2];

  n[0] = SegmentNodes[e*2];
  n[1] = SegmentNodes[e*2+1];

  Triangle<SPD>::map(&Triangle<SPD>::ParamCoord[2*n[0]], NodalCoord );
  Triangle<SPD>::map(&Triangle<SPD>::ParamCoord[2*n[1]], NodalCoord+2);

  v[0] = NodalCoord[2]-NodalCoord[0];
  v[1] = NodalCoord[3]-NodalCoord[1];

  double norm = sqrt(v[0]*v[0]+v[1]*v[1]);

  if(norm<=0) {
    std::cerr << 
      "The normal cannot be computed. Two vertices of a polytope seem to coincide\n";
  }

  VNormal.push_back( v[1]/norm);
  VNormal.push_back(-v[0]/norm);
}

/**
 * computes the center of the in-circle of a triangle
 * by computing the point of intersection of bisectors of the 
 * sides, which are perpendicular to the sides
 */
template <size_t SPD>
void Triangle<SPD>::computeCenter (VecDouble& center) const {

  double x1 = AbstractGeom<SPD>::getCoordinate (0, 0); // node 0, x coord
  double y1 = AbstractGeom<SPD>::getCoordinate (0, 1); // node 0, y coord

  double x2 = AbstractGeom<SPD>::getCoordinate (1, 0); // node 0, y coord
  double y2 = AbstractGeom<SPD>::getCoordinate (1, 1); // node 0, y coord


  double x3 = AbstractGeom<SPD>::getCoordinate (2, 0); // node 0, y coord
  double y3 = AbstractGeom<SPD>::getCoordinate (2, 1); // node 0, y coord


  // check if the slope of some side will come out to inf
  // and swap with third side 
  if (fabs(x2 - x1) < TOLERANCE) { // almost zero
    std::swap (x2, x3);
    std::swap (y2, y3);
  } 

  if (fabs(x3 - x2) < TOLERANCE) {
    std::swap (x1, x2);
    std::swap (y1, y2);
  }


  // mid points of the sides
  double xb1 = midpoint(x1, x2);
  double yb1 = midpoint(y1, y2);

  double xb2 = midpoint(x2, x3);
  double yb2 = midpoint(y2, y3);

  double xb3 = midpoint(x3, x1);
  double yb3 = midpoint(y3, y1);

  // slopes of all sides
  double m1 = (y2 - y1) / (x2 - x1);
  double m2 = (y3 - y2) / (x3 - x2);
  double m3 = (y3 - y1) / (x3 - x1);

  // solve simultaneous equations for first 2 bisectors
  double cy = (xb2 - xb1 + m2 * yb2 - m1 * yb1) / (m2 - m1);
  double cx = (m2 * xb1 - m1 * xb2 + m2 * m1 * yb1 - m2 * m1 * yb2) / (m2 - m1);

  // check against the third bisector
  if (fabs(x3-x1) > 0) { // checks if m3 == inf
    assert(fabs((cx + m3 * cy) - (xb3 + m3 * yb3)) < 1e-9);
  }

  // output the computed values
  center[0] = cx;
  center[1] = cy;
}

#endif
