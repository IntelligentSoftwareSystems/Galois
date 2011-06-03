/*
 * main.cpp
 *
 *  Created on: Nov 11, 2010
 *      Author: amshali
 */

#include "Barneshut.h"
#include "Galois/Launcher.h"
#include "Galois/Runtime/Timer.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <sys/time.h>
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Barnshut N-Body Simulator";
static const char* description = "Simulation of the gravitational forces in a galactic cluster using the Barnes-Hut n-body algorithm\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/barneshut.html";
static const char* help = "file <input file> | gen <numbodies> <ntimesteps> <seed>";

Graph *octree;
GNode root;
int step;
Barneshut barneshut;

struct process {
  template<typename Context>
  void operator()(OctTreeNodeData& item, Context& lwl) {
    barneshut.computeForce(item, octree, root, barneshut.diameter,
			   barneshut.itolsq, step, barneshut.dthf, barneshut.epssq);
  }
};

void parse_args(const std::vector<const char*>& args, Barneshut& app) {
  int size = args.size();
  int index = 0;

  if (index + 2 >= size) {
    std::cerr << "not enough arguments, use -help for usage information\n";
    exit(1);
  } else if (std::string("file").compare(args[index]) == 0 && index + 1 < size) {
    app.readInput(args[index + 1]);
  } else if (std::string("gen").compare(args[index]) == 0 && index + 3 < size) {
    app.genInput(atoi(args[index+1]), atoi(args[index+2]), atoi(args[index+3]));
  } else {
    std::cerr << "wrong arguments, use -help for usage information\n";
    exit(1);
  }
}

OctTreeNodeData omain() {
	OctTreeNodeData res;
	for (step = 0; step < barneshut.ntimesteps; step++) {
		barneshut.computeCenterAndDiameter();

    octree = new Graph(OctTreeNodeData(barneshut.centerx, barneshut.centery, barneshut.centerz));
    root = octree;

    barneshut.insertPoints(octree, root);

    // summarize subtree info in each internal node
    // (plus restructure tree and sort bodies for performance reasons)
    barneshut.curr = 0;
		barneshut.computeCenterOfMass(octree, root);

    GaloisRuntime::WorkList::dChunkedLIFO<OctTreeNodeData, 256> wl;
    wl.fill_initial(&barneshut.leaf[0], &barneshut.leaf[barneshut.curr]);
    Galois::for_each(wl, process());

    // advance the position and velocity of each
		barneshut.advance(octree, barneshut.dthf, barneshut.dtime);

		if (Galois::Launcher::isFirstRun()) {
			res = root->getData(Galois::Graph::NONE);
			std::cout << "Timestep " << step << " Center of Mass = " << res.posx
					<< " " << res.posy << " " << res.posz << std::endl;
		}
		delete octree;
	} 
  return res;
}

void verify(OctTreeNodeData res, std::vector<const char*>& args) {
  Barneshut b2;
  parse_args(args, b2);
  OctTreeNodeData s_res;
  for (step = 0; step < b2.ntimesteps; step++) {
    b2.computeCenterAndDiameter();
    Graph* octree2 = new Graph(OctTreeNodeData(b2.centerx, b2.centery, b2.centerz));
    root = octree2;

    b2.insertPoints(octree2, root);
    b2.curr = 0;
    b2.computeCenterOfMass(octree2, root);

    for (int kk = 0; kk < b2.curr; kk++) {
      b2.computeForce(b2.leaf[kk], octree2, root,
          b2.diameter, b2.itolsq, step, b2.dthf,
          b2.epssq);
    }
    b2.advance(octree2, b2.dthf, b2.dtime);
    s_res = root->getData(Galois::Graph::NONE);
    delete octree2;
  }

  if ((fabs(res.posx - s_res.posx) / fabs(std::min(res.posx, s_res.posx))
      > 0.001) || (fabs(res.posy - s_res.posy) / fabs(std::min(res.posy,
      s_res.posy)) > 0.001) || (fabs(res.posz - s_res.posz) / fabs(std::min(
      res.posz, s_res.posz)) > 0.001)) {
    std::cerr << "verification failed" << std::endl;
  } else {
    std::cerr << "verification succeeded" << std::endl;
  }
}

int main(int argc, const char** argv) {
  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

  std::vector<const char*> args = parse_command_line(argc, argv, help);
  parse_args(args, barneshut);
  printBanner(std::cout, name, description, url);
  std::cerr << "configuration: "
            << barneshut.nbodies << " bodies, "
            << barneshut.ntimesteps << " time steps" << std::endl << std::endl;
	std::cout << "Num. of threads: " << numThreads << std::endl;

  Galois::Launcher::startTiming();
  OctTreeNodeData res = omain();
  Galois::Launcher::stopTiming();
  std::cout << "STAT: Time (without input gen) " << Galois::Launcher::elapsedTime() << "\n";
  if (Galois::Launcher::isFirstRun() && !skipVerify) { // verify result
    verify(res, args);
  }
}
