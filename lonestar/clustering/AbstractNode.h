/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef ABSTRACTNODE_H_
#define ABSTRACTNODE_H_

#include "Point3.h"

#include <assert.h>
#include <iostream>
#include <limits>
#include <stdlib.h>


class AbstractNode {
public:
  static int globalNumReps;
  static std::vector<std::vector<double>*> repRandomNums;
  static bool globalMultitime;

protected:
  Point3 myLoc;
  Point3 intensity; // Use r,g,b as x,y,z
  int startTime, endTime;
  GVector<double> timeVector;

public:
  AbstractNode(double x, double y, double z) : myLoc(x, y, z), intensity(0) {
    startTime = -1;
  }

  AbstractNode() : myLoc(0), intensity(0) { startTime = -1; }

  virtual ~AbstractNode() {}

  double getScalarTotalIntensity() const {
    return (1.0f / 3.0f) * intensity.getSum();
  }

  double getRelativeIntensity(int time) const {
    if (time < startTime || time > endTime)
      return 0;
    return timeVector[time - startTime];
  }

  void setIntensity(double inScaleFactor, int inTime) {

    intensity.set(inScaleFactor);

    if (inTime == -1) {
      inTime = 0;
    }
    if (inTime >= 0) {
      startTime = inTime;
      endTime   = inTime;
      timeVector.clear();
      timeVector.push_back(1.0f);
    } else {
      // negative value used as signal that should be uniform across all time
      int len   = -inTime;
      startTime = 0;
      endTime   = (len - 1);
      timeVector.clear();
      timeVector.resize(len);
      for (int i = 0; i < len; i++) {
        timeVector[i] = 1.0f / len;
      }
      scaleIntensity(len);
    }
  }
  void setSummedIntensity(AbstractNode& inA, AbstractNode& inB) {
    intensity.set(inA.intensity);
    intensity.add(inB.intensity);
    startTime = inA.startTime < inB.startTime ? inA.startTime : inB.endTime;
    endTime   = inA.startTime < inB.startTime ? inB.startTime : inA.endTime;

    if (startTime != endTime) {
      int len = endTime - startTime + 1;
      if ((timeVector.size() == 0) || timeVector.size() < (unsigned int)len) {
        timeVector.resize(len);
      } else {
        for (unsigned int i = 0; i < timeVector.size(); i++) {
          timeVector[i] = 0;
        }
      }
      double weightA  = inA.getScalarTotalIntensity();
      double weightB  = inB.getScalarTotalIntensity();
      double invDenom = 1.0f / (weightA + weightB);
      weightA *= invDenom;
      weightB *= invDenom;
      for (int i = inA.startTime; i <= inA.endTime; i++) {
        timeVector[i - startTime] +=
            weightA * inA.timeVector[i - inA.startTime];
      }
      for (int i = inB.startTime; i <= inB.endTime; i++) {
        timeVector[i - startTime] +=
            weightB * inB.timeVector[i - inB.startTime];
      }
    } else {
      timeVector.clear();
      timeVector.push_back(1.0f);
    }
  }

  void scaleIntensity(double inScale) { intensity.scale(inScale); }

  static void setGlobalNumReps() {
    if (globalNumReps == 1) {
      return;
    }
    // trees must be rebuilt for this to take effect
    globalNumReps = 1;
    double inc    = 1.0f / 1;
    for (int i = 0; i < 256; i++) {
      for (unsigned int i = 0; i < repRandomNums.size(); i++) {
        std::vector<double>* ranVec = new std::vector<double>(1);
        for (int j = ranVec->size() - 1; j > 0; j++) {
          int index = (int)(j + 1) * (inc * (double)rand()) /
                      (std::numeric_limits<int>::max());
          if (index > j) {
            GALOIS_DIE("Badness :", index);
          }
          double temp      = (*ranVec)[j];
          (*ranVec)[j]     = (*ranVec)[index];
          (*ranVec)[index] = temp;
        }
        if (AbstractNode::repRandomNums[i] != NULL)
          delete AbstractNode::repRandomNums[i];
        AbstractNode::repRandomNums[i] = ranVec;
      }
    }
  }

  static void setGlobalMultitime() {
    // trees must be rebuilt for this to take effect
    globalMultitime = false;
  }

  Point3& getPoint() { return myLoc; }

  virtual bool isLeaf() = 0;

  virtual int size() = 0;
  static void cleanup() {
    for (unsigned int i = 0; i < repRandomNums.size(); i++)
      delete AbstractNode::repRandomNums[i];
  }
  friend std::ostream& operator<<(std::ostream& s, AbstractNode& pt);
};

std::ostream& operator<<(std::ostream& s, AbstractNode& pt) {
  s << "Abs Node :: Loc " << pt.myLoc << " , Int ::" << pt.intensity
    << " Time:: [" << pt.startTime << " - " << pt.endTime << "]";
  return s;
}

const int numRepRandomNums = 256;
std::vector<std::vector<double>*> AbstractNode::repRandomNums(256);
int AbstractNode::globalNumReps    = -1;
bool AbstractNode::globalMultitime = false;

#endif /* ABSTRACTNODE_H_ */
