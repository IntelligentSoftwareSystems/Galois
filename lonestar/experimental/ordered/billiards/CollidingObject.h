/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef _COLLIDING_OBJECT_H_
#define _COLLIDING_OBJECT_H_

#include <string>
#include <iostream>

#include <cstdlib>
#include <cstdio>
#include <cmath>

class Event;

class CollidingObject {

public:
  virtual ~CollidingObject(void) {}

  virtual unsigned collCounter(void) const = 0;

  virtual void incrCollCounter(void) = 0;

  // need for objects that don't need to recompute their
  // collisions with other bodies. Usually, it's the moving bodies
  // i.e. balls that need to recompute their collisions with other
  // balls and cushions etc, while cushions and potted balls don't need to
  // do so.
  virtual bool isStationary(void) const = 0;

  virtual unsigned getID(void) const = 0;

  virtual std::string str(void) const = 0;

  virtual void simulate(const Event& e) = 0;
};

#endif // _COLLIDING_OBJECT_H_
