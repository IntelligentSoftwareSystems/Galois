/** Barnes-hut application -*- C++ -*-
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
 * @author Martin Burtscher <burtscher@txstate.edu>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <sstream>


#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "Galois/Galois.h"
#include "Galois/GaloisUnsafe.h"
#include "Galois/Atomic.h"
#include "Galois/Statistic.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

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
const char* desc = "Simulates the gravitational forces in a galactic cluster using the "
  "Barnes-Hut n-body algorithm";
const char* url = "barneshut";

namespace cll = llvm::cl;

static cll::opt<int> nbodies("n", cll::desc("Number of bodies"), cll::init(10000));
static cll::opt<int> ntimesteps("steps", cll::desc("Number of steps"), cll::init(1));
static cll::opt<int> seed("seed", cll::desc("Random seed"), cll::init(7));


enum TreeSummMethod {
  SERIAL, 
  SERIAL_TREE,
  GALOIS_TREE,
  KDG_HAND, 
  KDG_SEMI, 
  LEVEL_HAND, 
  SPEC, 
  TWO_PHASE, 
  LEVEL_EXEC, 
  CILK_EXEC
};

cll::opt<TreeSummMethod> treeSummOpt (
    cll::desc ("Tree Summarization Method:"),
    cll::values (
      clEnumVal (SERIAL, "Serial recursive"),
      clEnumVal (SERIAL_TREE, "using data dependence DAG version of KDG"),
      clEnumVal (GALOIS_TREE, "using data dependence DAG version of KDG"),
      clEnumVal (KDG_HAND, "KDG based hand-implemented"),
      clEnumVal (KDG_SEMI, "KDG based semi-automated"),
      clEnumVal (LEVEL_HAND, "level-by-level hand-implemented"),
      clEnumVal (SPEC, "using speculative ordered executor"),
      clEnumVal (TWO_PHASE, "using two phase window ordered executor"),
      clEnumVal (LEVEL_EXEC, "using level-by-level executor"),
      clEnumVal (CILK_EXEC, "using cilk executor"),
      clEnumValEnd),
    cll::init (SERIAL));

double nextDouble() {
  return rand() / (double) RAND_MAX;
}

/**
 * Generates random input according to the Plummer model, which is more
 * realistic but perhaps not so much so according to astrophysicists
 */
template <typename BodyCont>
void generateInput(BodyCont& bodies, int nbodies, int seed) {
  typedef typename std::remove_pointer<typename BodyCont::value_type>::type Body_ty;

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

    Body_ty* b = new Body_ty ();
    b->mass = 1.0 / nbodies;
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

template<bool IsOn>
struct ToggleTime: public Galois::StatTimer {
  ToggleTime(const char* name): Galois::StatTimer(name) { }
};

template<>
struct ToggleTime<false> {
  ToggleTime(const char* name) { }
  void start() { }
  void stop() { }
};


template <bool TrackTime, typename TB>
Point run(int nbodies, int ntimesteps, int seed, const TB& treeBuilder) {
  typedef typename TB::Base_ty B;
  typedef Galois::gdeque<Body<B>*> Bodies;
  typedef Galois::GFixedAllocator<OctreeInternal<B> > TreeAlloc;
  

  Config config;
  Bodies bodies;
  TreeAlloc treeAlloc;

  ToggleTime<TrackTime> t_input_gen ("Time taken by input generation: ");

  t_input_gen.start ();
  generateInput (bodies, nbodies, seed);
  t_input_gen.stop ();

  Point ret;
  for (int step = 0; step < ntimesteps; step++) {
    typedef Galois::WorkList::dChunkedLIFO<256> WL;


    BoundingBox box;
    ToggleTime<TrackTime> t_bbox ("Time taken by Bounding Box computation: ");

    // TODO: use parallel reducer here
    struct GetPos {
      typedef const Point& result_type;

      result_type operator () (const Body<B>* b) const {
        return b->pos;
      }
    };
    auto beg = boost::make_transform_iterator (bodies.begin (), GetPos ());
    auto end = boost::make_transform_iterator (bodies.end (), GetPos ());

    t_bbox.start ();
    Galois::for_each(beg, end,
        ReduceBoxes<B>(box), Galois::wl<WL> ());
    t_bbox.stop ();


    // OctreeInternal<B>* top = new OctreeInternal<B>(box);
    ToggleTime<TrackTime> t_tree_build ("Time taken by Octree building and summarization: ");

    t_tree_build.start ();
    OctreeInternal<B>* top = treeBuilder (box, treeAlloc, bodies.begin (), bodies.end ());
    t_tree_build.stop ();

    if (!skipVerify) {
      BuildSummarizeSeparate<BuildSummarizeSerial<SerialNodeBase>, SummarizeTreeSerial> serialBuilder;
      OctreeInternal<SerialNodeBase>* stop = serialBuilder (box, treeAlloc, bodies.begin (), bodies.end ());

      compareTrees (stop, top);
    }


    // BuildOctreeSerial<B> build;
// 
    // OctreeInternal<B>* top = build (box, bodies.begin (), bodies.end ());
// 
    // // Galois::for_each(bodies.begin(), bodies.end(),
        // // BuildOctree<B>(top, box.radius()), Galois::wl<WL> ());
// 
    // // reset the number of threads
    // Galois::setActiveThreads(numThreads);
// 
    // ToggleTime<TrackTime> t_tree_summ ("Time taken by Tree Summarization: ");
// 
    // t_tree_summ.start ();
    // summMethod (top, bodies.begin (), bodies.end ());
    // t_tree_summ.stop ();


    if (false) { // disabling remaining phases
      ToggleTime<TrackTime> T_parallel("ParallelTime");
      T_parallel.start();

      Galois::for_each(bodies.begin(), bodies.end(),
          ComputeForces<B> (config, top, box.diameter()), Galois::wl<WL> ());
      Galois::for_each(bodies.begin(),bodies.end(),
          AdvanceBodies<B> (config), Galois::wl<WL> ());
      T_parallel.stop();
    }

    ret = top->pos;

    std::cout 
      << "Timestep " << step
      << ", Root's Center of Mass = " << top->pos << std::endl;

    // TODO: delete using TreeAlloc
    // delete top;

    for (auto i = bodies.begin (), endi = bodies.end ();
        i != endi; ++i) {
      delete *i;
      *i = nullptr;
    }

  }

  return ret;
}

} // end namespace bh

int main(int argc, char** argv) {
  Galois::StatManager sm;
  LonestarStart(argc, argv, bh::name, bh::desc, bh::url);

  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

  std::cerr << "configuration: "
            << bh::nbodies << " bodies, "
            << bh::ntimesteps << " time steps" << std::endl << std::endl;
  std::cout << "Num. of threads: " << numThreads << std::endl;

  bh::Point pos;
  Galois::StatTimer T;

  T.start();
  switch (bh::treeSummOpt) {
    case bh::SERIAL:
      // TODO: fix template argument mistmatch between build and summarize
      pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::BuildSummarizeSeparate<bh::BuildTreeSerial<bh::SerialNodeBase>, bh::SummarizeTreeSerial>  ());
      break;
      
    // case bh::SERIAL_TREE:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::BuildSummarizeSerial ());
      // break;
      // 
    // case bh::GALOIS_TREE:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::BuildSummarizeGalois ());
      // break;
// 

    // case bh::KDG_HAND:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::TreeSummarizeODG ());
      // break;
// 
    // case bh::KDG_SEMI:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::TreeSummarizeKDGsemi ());
      // break;
// 
    // case bh::LEVEL_HAND:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::TreeSummarizeLevelByLevel ());
      // break;
// 
    // case bh::SPEC:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::TreeSummarizeSpeculative ());
      // break;
// 
    // case bh::TWO_PHASE:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::TreeSummarizeTwoPhase ());
      // break;
// 
    // case bh::LEVEL_EXEC:
      // // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::TreeSummarizeLevelExec ());
      // break;
// 
    // case bh::CILK_EXEC:
      // pos = bh::run<true> (bh::nbodies, bh::ntimesteps, bh::seed, bh::TreeSummarizeCilk ());
      // break;

    default:
      abort ();

  }
  T.stop();


  // if (!skipVerify) {
    // std::cout << "Running serial tree summarization for verification" << std::endl;
    // bh::Point serPos = bh::run<false> (bh::nbodies, bh::ntimesteps, bh::seed, bh::SummarizeTreeSerial ());
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
