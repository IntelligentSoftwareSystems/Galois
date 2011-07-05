#ifndef TETRAHEDRON
#define TETRAHEDRON

#include <algorithm>
#include <cassert>

#include "ElementGeometry.h"
#include "Triangle.h"

/**
   \brief Geometry of 3D tetrahedra.
   
   A tetrahedron is:
   1) A set of indices that describe the connectivity of the tetrahedran, properly oriented. 
   
   2) An affine map from a three-dimensional tetrahedron (parametric configuration) with 
   volume 1/6 to the 
   convex hull of 4 vertices. 
   
   The parametric configuration of the tetrahedron is 0(1,0,0), 1(0,1,0), 2(0,0,0), 3(0,0,1).
   The parametric coordinates are the ones associated with vertices 0,1 and 3.
   The faces (for the purpose of quadrature points) are ordered as:
   1) Face 1: 2-1-0,
   2) Face 2: 2-0-3,
   3) Face 3: 2-3-1,
   4) Face 4: 0-1-3.
   The convention used in numbering these faces is that the resulting normal is 
   always outward.

*/

#define TET_SPD 3

class Tetrahedron: public AbstractGeom<TET_SPD>
{
private:
  static const double ParamCoord[]; 
  static const size_t FaceNodes[];


public:
  
  //! Connectivity in Tetrahedron::GlobalCoordinatesArray
  Tetrahedron (const std::vector<double>& globalCoordVec, const std::vector<GlobalNodalIndex>& connectivity)
    :AbstractGeom<TET_SPD> (globalCoordVec, connectivity) {
      assert (connectivity.size () == 4);
  }


  inline virtual ~Tetrahedron(){}

  Tetrahedron(const Tetrahedron & that) : AbstractGeom<TET_SPD>(that) {
  }

  virtual Tetrahedron* clone() const {
    return new Tetrahedron(*this);
  }


  //! Returns the number of vertices.
  inline const size_t getNumVertices() const { return 4; }


  //! Returns the name of the geometry. It clarifies the meaning of the connectivity array.
  inline const std::string getPolytopeName() const { return "TETRAHEDRON"; }

  //! Returns the number of dimensions in the parametric configuartion.
  inline const size_t getParametricDimension() const { return 3; }

  //! Returns the number of dimensions in the real configuration.
  inline const size_t getEmbeddingDimension() const { return 3; }

  //! Number of faces the polytope has.
  inline size_t getNumFaces() const { return 4; }
  
  //! map from parametric to real configuration.
  //! \param X parametric coordinates.
  //! \param Y returned real coordinates.
  void map(const double *X, double *Y) const {
    const size_t sd = AbstractGeom<TET_SPD>::getSpatialDimension ();

    for(size_t i=0; i<sd; i++)
      Y[i] = 
        X[0]*AbstractGeom<TET_SPD>::getCoordinate(0,i) + 
        X[1]*AbstractGeom<TET_SPD>::getCoordinate(1,i) + 
        X[2]*AbstractGeom<TET_SPD>::getCoordinate(3,i) + 
        (1.0-X[0]-X[1]-X[2])*AbstractGeom<TET_SPD>::getCoordinate(2,i);
  }

  //! Derivative of the map from the parametric to the real configuration.
  //! \param X parametric cooridnates
  //! \param Jac returns Jacobian of the map.
  //! \param DY returnd derivative of the map. 
  //! Here DY[a*getEmbeddingDimension()+i] contains the derivative of the a-th direction of the i-th coordinate.
  void dMap(const double *X, double *DY, double &Jac) const {
    const size_t sd = AbstractGeom<TET_SPD>::getSpatialDimension (); // spatial_dimension.

    for(size_t i=0; i<sd; i++) // Loop over x,y,z
    {
      DY[i]      = AbstractGeom<TET_SPD>::getCoordinate(0,i)-AbstractGeom<TET_SPD>::getCoordinate(2,i);
      DY[sd*1+i] = AbstractGeom<TET_SPD>::getCoordinate(1,i)-AbstractGeom<TET_SPD>::getCoordinate(2,i);
      DY[sd*2+i] = AbstractGeom<TET_SPD>::getCoordinate(3,i)-AbstractGeom<TET_SPD>::getCoordinate(2,i);
    }

    Jac = 0.;
    for(size_t i=0; i<sd; i++)
    {
      size_t i1 = (i+1)%sd;
      size_t i2 = (i+2)%sd;
      Jac += DY[i]*(DY[1*sd+i1]*DY[2*sd+i2] - DY[2*sd+i1]*DY[1*sd+i2]);
    }

    Jac = fabs(Jac);
  } 

  //! Creates a new ElementGeometry object corresponding to face 'e' in the polytope. The object ghas to be 
  //! deleted by the recepient.
  //! \param e facenumber , starting from 0.
  //! Prompts an error if an invalid face is requested.
  Triangle<3> * getFaceGeometry(size_t e) const {

    if(e>=0 && e<=3) {
      std::vector<GlobalNodalIndex> conn(FaceNodes + 3*e, FaceNodes + 3*e + 2);

      return new Triangle<TET_SPD> (AbstractGeom<TET_SPD>::getGlobalCoordVec (), conn);

      // return new Triangle<3>(AbstractGeom<TET_SPD>::getConnectivity()[FaceNodes[3*e+0]], 
      // AbstractGeom<TET_SPD>::getConnectivity()[FaceNodes[3*e+1]], 
      // AbstractGeom<TET_SPD>::getConnectivity()[FaceNodes[3*e+2]]);
    }
    else {
      std::cout<<"\nTetrahedron::GetFaceGeometry() : Request for invalid face. Quitting...\n";
      exit(1);
    }
  }

  //! Get the inradius
  const double getInRadius(void) const {
  double a[3],b[3],c[3],o,t[3],d[3],t1[3],t2[3], t3[3];
  for(size_t i=0;i<3;i++) {
    o = AbstractGeom<TET_SPD>::getCoordinate(0,i);
    a[i]=AbstractGeom<TET_SPD>::getCoordinate(1,i) - o;
    b[i]=AbstractGeom<TET_SPD>::getCoordinate(2,i) - o;
    c[i]=AbstractGeom<TET_SPD>::getCoordinate(3,i) - o;
  } 
  cross(b,c,t);
  cross(c,a,d);
  d[0]+=t[0];
  d[1]+=t[1];
  d[2]+=t[2];
  cross(a,b,t);
  d[0]+=t[0];
  d[1]+=t[1];
  d[2]+=t[2];
  cross(b,c,t);
  cross(b,c,t1);
  cross(c,a,t2);
  cross(a,b,t3); 
  double rv = (dot(a,t)/(mag(t1)+mag(t2)+mag(t3)+ mag(d)));

  return(rv);
  }

  //! Get the outradius -- radius of the circumscribed sphere
  const double getOutRadius(void) const {
    double x[4],y[4],z[4],r2[4],ones[4]; 
    double M11, M12, M13, M14, M15;
    double **a;
    a = (double **) malloc(4*sizeof(double *));

    for(size_t i = 0; i < 4; i++) {
      x[i] = AbstractGeom<TET_SPD>::getCoordinate(i,0);
      y[i] = AbstractGeom<TET_SPD>::getCoordinate(i,1);
      z[i] = AbstractGeom<TET_SPD>::getCoordinate(i,2);
      r2[i] = x[i]*x[i] + y[i]*y[i] + z[i]*z[i];
      ones[i] = 1.0;
    } 
    a[0] = x;
    a[1] = y;
    a[2] = z;
    a[3] = ones;
    M11 = determinant(a,4);
    a[0] = r2;
    M12 = determinant(a,4);
    a[1] = x;
    M13 = determinant(a,4);
    a[2] = y;
    M14 = determinant(a,4);
    a[3] = z;
    M15 = determinant(a,4);

    double x0,y0,z0;
    x0 = 0.5 * M12/M11;
    y0 = -0.5 * M13/M11;
    z0 = 0.5 * M14/M11;

    free(a);
    return(sqrt(x0*x0 + y0*y0 + z0*z0 - M15/M11));
  }


  //! Compute external normal for a face
  //!
  //! @param e: face number for which the normal is desired
  //! @param vNormal: output of the three Cartesian components of the normal vector
  virtual void computeNormal (size_t e, std::vector<double>& vNormal) const {
    const size_t sd = getSpatialDimension();

    size_t n0, n1, n2;   // Local node numbers of face 'e'

    n0 = FaceNodes[3*e]; n1 = FaceNodes[3*e+1]; n2 = FaceNodes[3*e+2];

    // Finding the coordinates of each node of face 'e':
    double p0[sd], p1[sd], p2[sd];

    map(&ParamCoord[3*n0], p0);
    map(&ParamCoord[3*n1], p1);
    map(&ParamCoord[3*n2], p2);

    double L01[sd];
    double L02[sd];
    for(size_t k=0; k<sd; k++)
    {
      L01[k] = p1[k]-p0[k];
      L02[k] = p2[k]-p0[k];
    }

    if(vNormal.size() < sd) vNormal.resize(sd);

    double vnorm2=0.;
    for(size_t k=0; k<sd; k++)
    {
      size_t n1 = (k+1)%sd;
      size_t n2 = (k+2)%sd;
      vNormal[k] = L01[n1]*L02[n2] - L01[n2]*L02[n1];
      vnorm2 += vNormal[k]*vNormal[k];
    }

    for(size_t k=0; k<sd; k++)
      vNormal[k] /= sqrt(vnorm2);

  }

protected:
static double mag(double *a) {
  double rv = sqrt(dot(a,a));
  return(rv); 
};

static double dot(double *a,double *b) {
  return(a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
};

static void cross(double *a,double *b, double *rv) {
  rv[0] =  a[1]*b[2] - a[2]*b[1];
  rv[1] =  a[2]*b[0] - a[0]*b[2];
  rv[2] =  a[0]*b[1] - a[1]*b[0];
};

static double determinant(double **a,size_t n) {
  // TODO: replace malloc with new

  size_t i,j,j1,j2;
  double det = 0;
  double **m = NULL;

  if (n < 1) { /* Error */

  } else if (n == 1) { /* Shouldn't get used */
    det = a[0][0];
  } else if (n == 2) {
    det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
  } else {
    det = 0;
    for (j1=0;j1<n;j1++) {
      m = (double**) malloc((n-1)*sizeof(double *));
      for (i=0;i<n-1;i++) {
        m[i] = (double *) malloc((n-1)*sizeof(double));
      }
      for (i=1;i<n;i++) {
        j2 = 0;
        for (j=0;j<n;j++) {
          if (j == j1) {
            continue;
          }
          m[i-1][j2] = a[i][j];
          j2++;
        }
      }
      det += pow(-1.0,1.0+j1+1.0) * a[0][j1] * determinant(m,n-1);
      for (i=0;i<n-1;i++) {
        free(m[i]);
      }
      free(m);
    }
  }
  return(det);
}


};

#endif
