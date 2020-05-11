#ifndef GALOIS_EDGEDATA_H
#define GALOIS_EDGEDATA_H

#include "Coordinates.h"
#include "NodeData.h"

class EdgeData {
private:
  bool border; //!< tells if this is a border edge
  double length; //!< length
  Coordinates middlePoint; //!< point at middle of this edge

public:
  //! default: not a border, negative length, no middle point
  EdgeData() : border(false), length(-1), middlePoint(){};

  //! Initialize all fields
  EdgeData(bool border, double length, Coordinates middlePoint)
      : border(border), length(length), middlePoint(middlePoint) {}

  // self explanatory functions below

  bool isBorder() const { return border; }

  void setBorder(bool isBorder) { EdgeData::border = isBorder; }

  double getLength() const { return length; }

  void setLength(double l) { EdgeData::length = l; }

  const Coordinates& getMiddlePoint() const { return middlePoint; }

  //! Explicitly set middle point given coordinate class
  void setMiddlePoint(const Coordinates& coordinates) {
    EdgeData::middlePoint.setXYZ(coordinates.getX(), coordinates.getY(),
                                 coordinates.getZ());
  }

  //! Explicitly set middle point given coordinates as three vars
  void setMiddlePoint(const double x, const double y, const double z) {
    EdgeData::middlePoint.setXYZ(x, y, z);
  }
};
#endif // GALOIS_EDGEDATA_H
