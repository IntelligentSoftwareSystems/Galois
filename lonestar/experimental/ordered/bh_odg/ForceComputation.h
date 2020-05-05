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

#ifndef BH_FORCE_COMPUTATION_H
#define BH_FORCE_COMPUTATION_H

#include "Config.h"
#include "Point.h"
#include "Octree.h"

namespace bh {

template <typename B>
struct ComputeForces {
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_aborts;

  Config& config;
  OctreeInternal<B>* top;
  double diameter;
  double root_dsq;

  ComputeForces(Config& config, OctreeInternal<B>* _top, double _diameter)
      : config(config), top(_top), diameter(_diameter) {
    root_dsq = diameter * diameter * config.itolsq;
  }

  template <typename Context>
  void operator()(Body<B>* bb, Context&) {
    Body<B>& b = *bb;
    Point p    = b.acc;
    for (int i = 0; i < 3; i++)
      b.acc[i] = 0;
    // recurse(b, top, root_dsq);
    iterate(b, root_dsq);
    for (int i = 0; i < 3; i++)
      b.vel[i] += (b.acc[i] - p[i]) * config.dthf;
  }

  void recurse(Body<B>& b, Body<B>* node, double dsq) {
    Point p;
    for (int i = 0; i < 3; i++)
      p[i] = node->pos[i] - b.pos[i];

    double psq = p.x * p.x + p.y * p.y + p.z * p.z;
    psq += config.epssq;
    double idr = 1 / sqrt(psq);
    // b.mass is fine because every body has the same mass
    double nphi  = b.mass * idr;
    double scale = nphi * idr * idr;
    for (int i = 0; i < 3; i++)
      b.acc[i] += p[i] * scale;
  }

  struct Frame {
    double dsq;
    OctreeInternal<B>* node;
    Frame(OctreeInternal<B>* _node, double _dsq) : dsq(_dsq), node(_node) {}
  };

  void iterate(Body<B>& b, double root_dsq) {
    std::vector<Frame> stack;
    stack.push_back(Frame(top, root_dsq));

    Point p;
    while (!stack.empty()) {
      Frame f = stack.back();
      stack.pop_back();

      for (int i = 0; i < 3; i++)
        p[i] = f.node->pos[i] - b.pos[i];

      double psq = p.x * p.x + p.y * p.y + p.z * p.z;
      if (psq >= f.dsq) {
        // Node is far enough away, summarize contribution
        psq += config.epssq;
        double idr   = 1 / sqrt(psq);
        double nphi  = f.node->mass * idr;
        double scale = nphi * idr * idr;
        for (int i = 0; i < 3; i++)
          b.acc[i] += p[i] * scale;

        continue;
      }

      double dsq = f.dsq * 0.25;

      for (int i = 0; i < 8; i++) {
        Octree<B>* next = f.node->getChild(i);
        if (next == NULL)
          break;
        if (next->isLeaf()) {
          // Check if it is me
          if (&b != next) {
            recurse(b, static_cast<Body<B>*>(next), dsq);
          }
        } else {
          stack.push_back(Frame(static_cast<OctreeInternal<B>*>(next), dsq));
        }
      }
    }
  }

  void recurse(Body<B>& b, OctreeInternal<B>* node, double dsq) {
    Point p;

    for (int i = 0; i < 3; i++)
      p[i] = node->pos[i] - b.pos[i];
    double psq = p.x * p.x + p.y * p.y + p.z * p.z;
    if (psq >= dsq) {
      // Node is far enough away, summarize contribution
      psq += config.epssq;
      double idr   = 1 / sqrt(psq);
      double nphi  = node->mass * idr;
      double scale = nphi * idr * idr;
      for (int i = 0; i < 3; i++)
        b.acc[i] += p[i] * scale;

      return;
    }

    dsq *= 0.25;

    for (int i = 0; i < 8; i++) {
      Octree<B>* next = node->child[i];
      if (next == NULL)
        break;
      if (next->isLeaf()) {
        // Check if it is me
        if (&b != next) {
          recurse(b, static_cast<Body<B>*>(next), dsq);
        }
      } else {
        recurse(b, static_cast<OctreeInternal<B>*>(next), dsq);
      }
    }
  }
};

template <typename B>
struct AdvanceBodies {
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_aborts;

  Config& config;

  explicit AdvanceBodies(Config& config) : config(config) {}

  template <typename Context>
  void operator()(Body<B>* bb, Context&) {
    Body<B>& b = *bb;
    Point dvel(b.acc);
    dvel *= config.dthf;

    Point velh(b.vel);
    velh += dvel;

    for (int i = 0; i < 3; ++i)
      b.pos[i] += velh[i] * config.dtime;
    for (int i = 0; i < 3; ++i)
      b.vel[i] = velh[i] + dvel[i];
  }
};

} // end namespace bh
#endif // BH_FORCE_COMPUTATION_H
