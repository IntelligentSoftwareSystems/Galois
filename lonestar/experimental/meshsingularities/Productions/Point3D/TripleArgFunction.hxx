#ifndef __TRIPLEARGFUNCTION_H_INCLUDED
#define __TRIPLEARGFUNCTION_H_INCLUDED
#include "EPosition.hxx"
#include "NPosition.hxx"
#include "../MatrixGeneration/IFunction.hxx"
#include <stdarg.h>
#include <vector>
#include <stdio.h>
namespace D3 {

double get_chi1(double var);
double get_chi2(double var);
double get_chi3(double var);

class ITripleArgFunction : public IFunction {

public:
  virtual double ComputeValue(double x, double y, double z) = 0;

  ITripleArgFunction(double* coordinates, bool* neighbours)
      : IFunction(coordinates, neighbours) {}

  ITripleArgFunction() {}

  virtual ~ITripleArgFunction() {}
};

class TripleArgFunctionWrapper : public ITripleArgFunction {
private:
  double (*f)(int, ...);

public:
  virtual double ComputeValue(double x, double y, double z) {
    return (*f)(3, x, y, z);
  }

  TripleArgFunctionWrapper(double (*f)(int, ...)) : f(f) {}

  virtual ~TripleArgFunctionWrapper() {}
};

class ShapeFunction : public ITripleArgFunction {
protected:
  double xl;
  double yl;
  double xr;
  double yr;
  double zl;
  double zr;
  EPosition position;

  double getXValueOnElement(double x) { return (x - xl) / (xr - xl); }

  double getYValueOnElement(double y) { return (y - yl) / (yr - yl); }

  double getZValueOnElement(double z) { return (z - zl) / (zr - zl); }

public:
  ShapeFunction(double* coordinates, bool* neighbours, EPosition position)
      : ITripleArgFunction(coordinates, neighbours), position(position) {
    xl = coordinates[0];
    xr = coordinates[1];
    yl = coordinates[2];
    yr = coordinates[3];
    zl = coordinates[4];
    zr = coordinates[5];
  }
};

class VertexBotLeftNearShapeFunction : public ShapeFunction {
public:
  VertexBotLeftNearShapeFunction(double* coordinates, bool* neighbours,
                                 EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);

  ~VertexBotLeftNearShapeFunction() {}
};

class VertexBotLeftFarShapeFunction : public ShapeFunction {
public:
  VertexBotLeftFarShapeFunction(double* coordinates, bool* neighbours,
                                EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);

  ~VertexBotLeftFarShapeFunction() {}
};

class VertexTopLeftNearShapeFunction : public ShapeFunction {
public:
  VertexTopLeftNearShapeFunction(double* coordinates, bool* neighbours,
                                 EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~VertexTopLeftNearShapeFunction() {}
};

class VertexTopLeftFarShapeFunction : public ShapeFunction {
public:
  VertexTopLeftFarShapeFunction(double* coordinates, bool* neighbours,
                                EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~VertexTopLeftFarShapeFunction() {}
};

class VertexTopRightNearShapeFunction : public ShapeFunction {
public:
  VertexTopRightNearShapeFunction(double* coordinates, bool* neighbours,
                                  EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  virtual ~VertexTopRightNearShapeFunction() {}
};

class VertexTopRightFarShapeFunction : public ShapeFunction {
public:
  VertexTopRightFarShapeFunction(double* coordinates, bool* neighbours,
                                 EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  virtual ~VertexTopRightFarShapeFunction() {}
};

class VertexBotRightNearShapeFunction : public ShapeFunction {
public:
  VertexBotRightNearShapeFunction(double* coordinates, bool* neighbours,
                                  EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~VertexBotRightNearShapeFunction() {}
};

class VertexBotRightFarShapeFunction : public ShapeFunction {
public:
  VertexBotRightFarShapeFunction(double* coordinates, bool* neighbours,
                                 EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~VertexBotRightFarShapeFunction() {}
};

class EdgeBotLeftShapeFunction : public ShapeFunction {
public:
  EdgeBotLeftShapeFunction(double* coordinates, bool* neighbours,
                           EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeBotLeftShapeFunction() {}
};

class EdgeTopLeftShapeFunction : public ShapeFunction {
public:
  EdgeTopLeftShapeFunction(double* coordinates, bool* neighbours,
                           EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeTopLeftShapeFunction() {}
};

class EdgeLeftNearShapeFunction : public ShapeFunction {
public:
  EdgeLeftNearShapeFunction(double* coordinates, bool* neighbours,
                            EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeLeftNearShapeFunction() {}
};

class EdgeLeftFarShapeFunction : public ShapeFunction {
public:
  EdgeLeftFarShapeFunction(double* coordinates, bool* neighbours,
                           EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeLeftFarShapeFunction() {}
};

class EdgeTopNearShapeFunction : public ShapeFunction {
public:
  EdgeTopNearShapeFunction(double* coordinates, bool* neighbours,
                           EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeTopNearShapeFunction() {}
};

class EdgeTopFarShapeFunction : public ShapeFunction {
public:
  EdgeTopFarShapeFunction(double* coordinates, bool* neighbours,
                          EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeTopFarShapeFunction() {}
};

class EdgeBotNearShapeFunction : public ShapeFunction {
public:
  EdgeBotNearShapeFunction(double* coordinates, bool* neighbours,
                           EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeBotNearShapeFunction() {}
};

class EdgeBotFarShapeFunction : public ShapeFunction {
public:
  EdgeBotFarShapeFunction(double* coordinates, bool* neighbours,
                          EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeBotFarShapeFunction() {}
};

class EdgeBotRightShapeFunction : public ShapeFunction {
public:
  EdgeBotRightShapeFunction(double* coordinates, bool* neighbours,
                            EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeBotRightShapeFunction() {}
};

class EdgeTopRightShapeFunction : public ShapeFunction {
public:
  EdgeTopRightShapeFunction(double* coordinates, bool* neighbours,
                            EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeTopRightShapeFunction() {}
};

class EdgeRightNearShapeFunction : public ShapeFunction {
public:
  EdgeRightNearShapeFunction(double* coordinates, bool* neighbours,
                             EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeRightNearShapeFunction() {}
};

class EdgeRightFarShapeFunction : public ShapeFunction {
public:
  EdgeRightFarShapeFunction(double* coordinates, bool* neighbours,
                            EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~EdgeRightFarShapeFunction() {}
};

class FaceLeftShapeFunction : public ShapeFunction {
public:
  FaceLeftShapeFunction(double* coordinates, bool* neighbours,
                        EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~FaceLeftShapeFunction() {}
};

class FaceRightShapeFunction : public ShapeFunction {
public:
  FaceRightShapeFunction(double* coordinates, bool* neighbours,
                         EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~FaceRightShapeFunction() {}
};

class FaceTopShapeFunction : public ShapeFunction {
public:
  FaceTopShapeFunction(double* coordinates, bool* neighbours,
                       EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~FaceTopShapeFunction() {}
};

class FaceBotShapeFunction : public ShapeFunction {
public:
  FaceBotShapeFunction(double* coordinates, bool* neighbours,
                       EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~FaceBotShapeFunction() {}
};

class FaceFarShapeFunction : public ShapeFunction {
public:
  FaceFarShapeFunction(double* coordinates, bool* neighbours,
                       EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~FaceFarShapeFunction() {}
};

class FaceNearShapeFunction : public ShapeFunction {
public:
  FaceNearShapeFunction(double* coordinates, bool* neighbours,
                        EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~FaceNearShapeFunction() {}
};

class InteriorShapeFunction : public ShapeFunction {
public:
  InteriorShapeFunction(double* coordinates, bool* neighbours,
                        EPosition position)
      : ShapeFunction(coordinates, neighbours, position) {}

  virtual double ComputeValue(double x, double y, double z);
  ~InteriorShapeFunction() {}
};
} // namespace D3
#endif
