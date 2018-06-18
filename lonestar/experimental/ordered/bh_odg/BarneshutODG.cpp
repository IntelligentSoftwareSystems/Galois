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

#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <sstream>

#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "galois/Galois.h"
#include "galois/Atomic.h"
#include "galois/CilkInit.h"
#include "galois/Timer.h"
#include "galois/runtime/DoAllCoupled.h"
#include "galois/runtime/Profile.h"
#include "galois/substrate/CompilerSpecific.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "Config.h"
#include "Point.h"
#include "Octree.h"
#include "BoundingBox.h"
#include "TreeBuildSumm.h"
#include "ForceComputation.h"

namespace bh {
const char* name = "Barnshut N-Body Simulator";
const char* desc =
    "Simulates the gravitational forces in a galactic cluster using the "
    "Barnes-Hut n-body algorithm";
const char* url = "barneshut";

namespace cll = llvm::cl;

static cll::opt<int> nbodies("n", cll::desc("Number of bodies"),
                             cll::init(10000));
static cll::opt<int> ntimesteps("steps", cll::desc("Number of steps"),
                                cll::init(1));
static cll::opt<int> seed("seed", cll::desc("Random seed"), cll::init(7));

enum TreeSummMethod {
  SERIAL,
  SERIAL_TREE,
  GALOIS_TREE,
  CILK_TREE,
  KDG_HAND,
  LEVEL_EXEC,
  SPEC,
  TWO_PHASE,
  DATA_DAG,
  RSUMM_SERIAL,
  RSUMM_CILK,
  RSUMM_GALOIS,
};

cll::opt<TreeSummMethod> treeSummOpt(
    cll::desc("Tree Summarization Method:"),
    cll::values(
        clEnumVal(SERIAL, "Serial recursive"),
        clEnumVal(SERIAL_TREE, "using data dependence DAG version of KDG"),
        clEnumVal(GALOIS_TREE, "using data dependence DAG version of KDG"),
        clEnumVal(CILK_TREE, "using cilk executor"),
        clEnumVal(KDG_HAND, "KDG based hand-implemented"),
        clEnumVal(LEVEL_EXEC, "using level-by-level executor"),
        clEnumVal(SPEC, "using speculative ordered executor"),
        clEnumVal(TWO_PHASE, "using two phase window ordered executor"),
        clEnumVal(DATA_DAG, "Generate DAG using data dependences"),
        clEnumVal(RSUMM_SERIAL, "Build Lock Free, Summarize Recursive Serial"),
        clEnumVal(RSUMM_CILK, "Build Lock Free, Summarize Recursive Cilk"),
        clEnumVal(RSUMM_GALOIS, "Build Lock Free, Summarize Recursive Galois"),
        clEnumValEnd),
    cll::init(SERIAL));

double nextDouble() { return rand() / (double)RAND_MAX; }

/**
 * Generates random input according to the Plummer model, which is more
 * realistic but perhaps not so much so according to astrophysicists
 */
template <typename BodyCont>
void generateInput(BodyCont& bodies, int nbodies, int seed) {
  typedef
      typename std::remove_pointer<typename BodyCont::value_type>::type Body_ty;

  double v, sq, scale;
  Point p;
  double PI = boost::math::constants::pi<double>();

  srand(seed);

  double rsc = (3 * PI) / 16;
  double vsc = sqrt(1.0 / rsc);

  for (int body = 0; body < nbodies; body++) {
    double r = 1.0 / sqrt(pow(nextDouble() * 0.999, -2.0 / 3.0) - 1);
    do {
      for (int i = 0; i < 3; i++)
        p[i] = nextDouble() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);

    Body_ty* b = new Body_ty();
    b->mass    = 1.0 / nbodies;
    for (int i = 0; i < 3; i++)
      b->pos[i] = p[i] * scale;

    do {
      p.x = nextDouble();
      p.y = nextDouble() * 0.1;
    } while (p.y > p.x * p.x * pow(1 - p.x * p.x, 3.5));
    v = p.x * sqrt(2.0 / sqrt(1 + r * r));
    do {
      for (int i = 0; i < 3; i++)
        p[i] = nextDouble() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    for (int i = 0; i < 3; i++)
      b->vel[i] = p[i] * scale;

    bodies.push_back(b);
  }
}

template <bool IsOn>
struct ToggleTime : public galois::StatTimer {
  ToggleTime(const char* name) : galois::StatTimer(name) {}
};

template <>
struct ToggleTime<false> {
  ToggleTime(const char* name) {}
  void start() {}
  void stop() {}
};

template <bool TrackTime, typename TB>
Point run(int nbodies, int ntimesteps, int seed, const TB& treeBuilder) {
  typedef typename TB::Base_ty B;
  typedef galois::gdeque<Body<B>*> Bodies;
  typedef galois::FixedSizeAllocator<OctreeInternal<B>> TreeAlloc;

  Config config;
  Bodies bodies;
  TreeAlloc treeAlloc;

  ToggleTime<TrackTime> t_input_gen("Time taken by input generation: ");

  t_input_gen.start();
  generateInput(bodies, nbodies, seed);
  t_input_gen.stop();

  Point ret;
  for (int step = 0; step < ntimesteps; step++) {
    typedef galois::worklists::PerSocketChunkLIFO<256> WL;

    ReducibleBox bbox;
    ToggleTime<TrackTime> t_bbox("Time taken by Bounding Box computation: ");

    // TODO: use parallel reducer here
    struct GetPos {
      typedef const Point& result_type;

      result_type operator()(const Body<B>* b) const { return b->pos; }
    };
    auto beg = boost::make_transform_iterator(bodies.begin(), GetPos());
    auto end = boost::make_transform_iterator(bodies.end(), GetPos());

    t_bbox.start();
    galois::do_all(beg, end, ReduceBoxes(bbox), galois::steal());
    t_bbox.stop();

    BoundingBox box(bbox.reduce());

    // OctreeInternal<B>* top = new OctreeInternal<B>(box);
    ToggleTime<TrackTime> t_tree_build(
        "Time taken by Octree building and summarization: ");

    t_tree_build.start();
    OctreeInternal<B>* top =
        treeBuilder(box, bodies.begin(), bodies.end(), treeAlloc);
    t_tree_build.stop();

    if (!skipVerify) {
      std::cout << "WARNING: Comparing against serially built & summarized "
                   "tree..., timing may be off"
                << std::endl;
      BuildSummarizeSeparate<BuildTreeSerial, SummarizeTreeSerial<B>>
          serialBuilder;
      OctreeInternal<B>* stop =
          serialBuilder(box, bodies.begin(), bodies.end(), treeAlloc);

      compareTrees(stop, top);
    }

    // BuildOctreeSerial<B> build;
    //
    // OctreeInternal<B>* top = build (box, bodies.begin (), bodies.end ());
    //
    // // galois::for_each(bodies.begin(), bodies.end(),
    // // BuildOctree<B>(top, box.radius()), galois::wl<WL> ());
    //
    // // reset the number of threads
    // galois::setActiveThreads(numThreads);
    //
    // ToggleTime<TrackTime> t_tree_summ ("Time taken by Tree Summarization: ");
    //
    // t_tree_summ.start ();
    // summMethod (top, bodies.begin (), bodies.end ());
    // t_tree_summ.stop ();

    if (false) { // disabling remaining phases
      ToggleTime<TrackTime> T_parallel("ParallelTime");
      T_parallel.start();

      galois::for_each(bodies.begin(), bodies.end(),
                       ComputeForces<B>(config, top, box.diameter()),
                       galois::wl<WL>());
      galois::for_each(bodies.begin(), bodies.end(), AdvanceBodies<B>(config),
                       galois::wl<WL>());
      T_parallel.stop();
    }

    ret = top->pos;

    std::cout << "Timestep " << step << ", Root's Center of Mass = " << top->pos
              << std::endl;

    // TODO: delete using TreeAlloc
    // delete top;
    destroyTree(top, treeAlloc);

    for (auto i = bodies.begin(), endi = bodies.end(); i != endi; ++i) {
      delete *i;
      *i = nullptr;
    }
  }

  return ret;
}

} // end namespace bh

int main(int argc, char** argv) {
  galois::StatManager sm;
  LonestarStart(argc, argv, bh::name, bh::desc, bh::url);

  std::cout.setf(std::ios::right | std::ios::scientific | std::ios::showpoint);

  std::cout << "configuration: " << bh::nbodies << " bodies, " << bh::ntimesteps
            << " time steps" << std::endl
            << std::endl;
  std::cout << "Num. of threads: " << numThreads << std::endl;

  bh::Point pos;
  galois::StatTimer T("total time:");

  T.start();
  switch (bh::treeSummOpt) {
  case bh::SERIAL:
    // TODO: fix template argument mistmatch between build and summarize
    pos =
        bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                      bh::BuildSummarizeSeparate<bh::BuildTreeSerial,
                                                 bh::SummarizeTreeSerial<>>());
    break;

  case bh::SERIAL_TREE:
    pos =
        bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                      bh::BuildSummarizeRecursive<bh::recursive::USE_SERIAL>());
    break;

  case bh::GALOIS_TREE:
    pos =
        bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                      bh::BuildSummarizeRecursive<bh::recursive::USE_GALOIS>());
    break;

  case bh::CILK_TREE:
    // FIXME:      galois::CilkInit ();
    pos = bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                        bh::BuildSummarizeRecursive<bh::recursive::USE_CILK>());
    break;

  case bh::LEVEL_EXEC:
    pos =
        bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                      bh::BuildSummarizeSeparate<bh::BuildTreeLockFree,
                                                 bh::TreeSummarizeLevelExec>());
    break;

  case bh::KDG_HAND:
    pos = bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                        bh::BuildSummarizeSeparate<bh::BuildTreeLockFree,
                                                   bh::TreeSummarizeKDGhand>());
    break;

  case bh::SPEC:
    pos = bh::run<true>(
        bh::nbodies, bh::ntimesteps, bh::seed,
        bh::BuildSummarizeSeparate<bh::BuildTreeLockFree,
                                   bh::TreeSummarizeSpeculative>());
    break;

  case bh::TWO_PHASE:
    pos =
        bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                      bh::BuildSummarizeSeparate<bh::BuildTreeLockFree,
                                                 bh::TreeSummarizeTwoPhase>());
    break;

  case bh::DATA_DAG:
    pos = bh::run<true>(bh::nbodies, bh::ntimesteps, bh::seed,
                        bh::BuildSummarizeSeparate<bh::BuildTreeLockFree,
                                                   bh::TreeSummarizeDataDAG>());
    break;

  case bh::RSUMM_SERIAL:
    pos = bh::run<true>(
        bh::nbodies, bh::ntimesteps, bh::seed,
        bh::BuildLockFreeSummarizeRecursive<bh::recursive::USE_SERIAL>());
    break;

  case bh::RSUMM_CILK:
    // FIXME:      galois::CilkInit ();
    pos = bh::run<true>(
        bh::nbodies, bh::ntimesteps, bh::seed,
        bh::BuildLockFreeSummarizeRecursive<bh::recursive::USE_CILK>());
    break;

  case bh::RSUMM_GALOIS:
    pos = bh::run<true>(
        bh::nbodies, bh::ntimesteps, bh::seed,
        bh::BuildLockFreeSummarizeRecursive<bh::recursive::USE_GALOIS>());
    break;

  default:
    abort();
  }
  T.stop();

  // if (!skipVerify) {
  // std::cout << "Running serial tree summarization for verification" <<
  // std::endl; bh::Point serPos = bh::run<false> (bh::nbodies, bh::ntimesteps,
  // bh::seed, bh::SummarizeTreeSerial ());
  //
  // double EPS = 1e-9;
  // bool equal = true;
  // for (unsigned i = 0; i < 3; ++i) {
  // if (fabs (pos[i] - serPos[i]) > EPS) {
  // equal = false;
  // break;
  // }
  // }
  //
  // if (!equal) {
  // std::cerr << "!!!BAD: Results don't match with serial!!!" << std::endl;
  // abort ();
  //
  // } else {
  // std::cout << ">>> OK, results verified" << std::endl;
  // }
  //
  // }
}
