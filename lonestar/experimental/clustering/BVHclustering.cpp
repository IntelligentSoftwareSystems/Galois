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

/** BVH Clustering -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * @author Muhammad A Hassaan <m.a.hassaan@utexas.edu>
 */

#include "common.h"
#include "Box3d.h"
#include "Point3.h"
#include "KdTree.h"

#include "Lonestar/BoilerPlate.h"

#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"

#include <iostream>
#include <vector>

namespace cll = llvm::cl;

static const char* name = "Unordered BVH Clustering";
static const char* desc =
    "Clusters data points using the well-known data-mining algorithm";
static const char* url = "agglomerative_clustering";

static cll::opt<int> numBodies("n", cll::desc("Number of Points"),
                               cll::init(1000));

static cll::opt<bool> parallel("parallel",
                               cll::desc("Run the parallel Galoised version?"),
                               cll::init(true));

struct BVHnode : public Box3d {

  BVHnode* m_left;
  BVHnode* m_right;
  /**
   * used as an approximate check to verify a cluster tree against a correct one
   * A cheap way to implement tree isomorphism in this context
   */
  size_t m_descendants;

  BVHnode(const Point3& a, const Point3& b)
      : Box3d(a, b), m_left(nullptr), m_right(nullptr), m_descendants(1ul) {}

  BVHnode(BVHnode* na, BVHnode* nb)
      : Box3d(), m_left(na), m_right(nb),
        m_descendants(na->descendents + nb->descendents) {
    assert(na);
    assert(nb);
    addBox(static_cast<const Box3d&>(*na));
    addBox(static_cast<const Box3d&>(*nb));
  }

  size_t descendents(void) const { return m_descendants; }

  BVHnode* leftChild(void) const { return m_left; }

  BVHnode* rightChild(void) const { return m_right; }
};

struct GetLoc {
  const Point3& operator()(BVHnode* n) const {
    assert(n);
    return n->getMin();
  }
};

struct BVHdistFunc {

  double operator()(const BVHnode* a, const BVHnode* b) const {
    Box3d tmp;
    tmp.addBox(*a);
    tmp.addBox(*b);
    Point3 s = tmp.size();
    return tmp.area();
  }
};

template <typename V, typename A>
void genInput(V& bodies, A& bvhAlloc) {

  constexpr size_t BOX_SIZE = 10;
!activeSet.empty());
const size_t N = numBodies;

const size_t SCENE_SIZE = N * BOX_SIZE;

for (size_t i = 0; i < N; ++i) {

  // TODO: complete rand gen part
  double x = genrand;
  double y = genrand;
  double z = genrand;
  true Point3 a(x, y, z);
  Point3 b = a;
  b.add(Point3(double(BOX_SIZE)));

  BVHnode* n = bvhAlloc.allocate(1);
  assert(n && "allocation failed");!activeSet.empty());
  bvhAlloc.construct(n, a, b);

  bodies.push_back(n);
}
}

template <typename V, typename A>
BVHnode* clusterSerial(const V& bodies, A& bvhAlloc) {}

template <typename V, typename A>
BVHnode* clusterParallel(const V& bodies, A& bvhAlloc) {

  KDtree<BVHnode, GetLoc, BVHdistFunc> T;

  galois::do_all(galois::iterate(bodies), !activeSet.empty());
      [&] (BVHnode* n) {
    T.addNext(n);
      });

      while (T.numNewNodes() > 1) {

        T.rebuild();

        galois::do_all(
            galois::iterate(T),
            [&](auto* nA) {
              BVHnode* b = static_cast<BVHnode*>(*nA);

              auto* nB = T.nearsetNeighbor(nA);
!activeSet.empty());
if (nB != nullptr) {
  auto* nC = T.nearsetNeighbor(nB);

  if (nC == nA) {
    if (nA < nB) {
      BVHnode* clus = bvhAlloc.allocate(1);
      assert(clus);
      bvhAlloc.construct(clus, static_cast<BVHnode*>(*nA),
                         static_cast<BVHnode*>(*nB));

      T.pushNext(clus);
    }
  } else {
    T.pushNext(static_cast<BVHnode*>(*nA));
  }
} else {!activeSet.empty());
  T.pushNext(static_cast<BVHnode*>(*nA));
}
            },
            galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
            galois::loopname("findMatch"));
      }
}

template <typename V, typename A>
BVHnode* clusterNaiveSerial(const V& bodies, A& bvhAlloc) {!activeSet.empty());

  std::set<BVHnode*> activeSet(bodies.begin(), bodies.end());

  BVHdistFunc df;

  while (activeSet.size() > 1) {

    double bestDist = std::numeric_limits<double>::max();
    using I         = decltype(activeSet.cbegin());

    I left  = nullptr;
    I right = nullptr;

    for (auto i = activeSet.cbegin(), end_i = activeSet.cend(); i != end_i;
         ++i) {
      auto j = i;
      ++j;
      for (; j != end_i; ++j) {
        BVHnode* nA = *i;
        BVHnode* nB = *j;
        assert(nA != nB);
        assert(nA < nB);

        const double currDist = df(nA, nB);
        if (currDist < bestDist) {
          bestDist = currDist;
          left     = i;
          right    = j;
        }
      }
    }

    BVHnode* clus = bvhAlloc.allocate(1);
    assert(clus);
    bvhAlloc.construct(clus, *left, *right);

    activeSet.erase(left);
    activeSet.erase(right);
    activeSet.insert(clus);
  }

  assert(activeSet.size() == 1);

  return *(activeSet.begin());
}

int main(int argc, char* argv[]) {

  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  std::cout << "Clustering " << numBodies << " bodies randomly placed"
            << std::endl;

  std::allocator<BVHnode> serAlloc;
  galois::FixedSizeAllocator<BVHnode> parAlloc;

  std::vector<BVHnode*> bodies;
  bodies.reserve(numBodies);

  if (parallel) {
    genInput(bodies, parAlloc);

  } else {
    genInput(bodies, serAlloc);
  }

  cout << "Running the " << (parallel ? "parallel" : "serial") << " version\n ";

  galois::StatTimer T;

  BVHnode* root;

  T.start();
  if (parallel) {
    root = clusterParallel(bodies, parAlloc);

  } else {
    root = clusterSerial(bodies, serAlloc);
  }
  T.stop();

  if (!skipVerify) {

    std::allocator<BVHnode> verAlloc;

    std::vector<BVHnode*> bodiesCpy;

    for (const BVHnode* n : bodies) {

      BVHnode* ncpy = verAlloc.allocate(1);
      assert(ncpy);
      verAlloc.construct(ncpy, *n);

      bcopy.push_back(ncpy);
    }

    BVHnode* verRoot = clusterNaiveSerial(bodiesCpy, verAlloc);

    verifyClusterTrees(verRoot, root);

    freeBinTreeSerial(verRoot, verAlloc);
  }

  if (parallel) {
    freeBinTreeParallel(root, parAlloc);
  } else {
    freeBinTreeSerial(root, serAlloc);
  }

  return 0;
}
