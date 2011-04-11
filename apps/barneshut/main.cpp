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
static const char* help = "[file <input file>|gen <seed> <numbodies>]";

Graph *octree;
GNode root;
int step;
Barneshut barneshut;

struct process {
  template<typename Context>
  void operator()(GNode& item, Context& lwl) {
    barneshut.computeForce(item, octree, root, barneshut.diameter,
			   barneshut.itolsq, step, barneshut.dthf, barneshut.epssq);
  }
};

void readInput(std::vector<const char*>& args, Barneshut& app) {
  if (args.size() < 2) {
    std::cerr << "not enough arguments, use -help for usage information\n";
    exit(1);
  } else if (strcmp(args[0], "file") == 0) {
    app.readInput(args[1]);
  } else if (strcmp(args[0], "gen") == 0) {
    app.genInput(atoi(args[1]), atoi(args[2]));
  } else {
    std::cerr << "wrong arguments, use -help for usage information\n";
    exit(1);
  }
}

int main(int argc, const char** argv) {
  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

  std::vector<const char*> args = parse_command_line(argc, argv, help);
  readInput(args, barneshut);
  printBanner(std::cout, name, description, url);
  std::cerr << "configuration: "
            << barneshut.nbodies << " bodies, "
            << barneshut.ntimesteps << " time steps" << std::endl << std::endl;
	std::cout << "Num. of threads: " << numThreads <<std::endl;
	Galois::setMaxThreads(numThreads);

	Galois::Launcher::startTiming();
	OctTreeNodeData res;
	for (step = 0; step < barneshut.ntimesteps; step++) { // time-step the system
		barneshut.computeCenterAndDiameter();

    octree = new Graph;
		root = createNode(octree, OctTreeNodeData(barneshut.centerx,
				barneshut.centery, barneshut.centerz)); 
    octree->addNode(root);
		double radius = barneshut.diameter * 0.5;
		for (int i = 0; i < barneshut.nbodies; i++) {
			OctTreeNodeData &b = barneshut.body[i];
			barneshut.insert(octree, root, b, radius);
		}
    barneshut.curr = 0;
    // summarize subtree info in each internal node
    // (plus restructure tree and sort bodies for performance reasons)
		barneshut.computeCenterOfMass(octree, root);

		std::vector<GNode> wl;
		for (int ii = 0; ii < barneshut.curr; ii++) {
			wl.push_back(barneshut.leaf[ii]);
		}
		Galois::for_each(wl.begin(), wl.end(), process());

    // advance the position and velocity of each
		barneshut.advance(octree, barneshut.dthf, barneshut.dtime);

		if (Galois::Launcher::isFirstRun()) {
			res = root.getData(Galois::Graph::NONE);
			std::cout << "Timestep " << step << " Center of Mass = " << res.posx
					<< " " << res.posy << " " << res.posz << std::endl;
		}
		delete octree;
	} 
	Galois::Launcher::stopTiming();
	std::cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";

	if (Galois::Launcher::isFirstRun()) { // verify result
		Barneshut b2;
    readInput(args, b2);
		OctTreeNodeData s_res;
		for (step = 0; step < b2.ntimesteps; step++) {
			b2.computeCenterAndDiameter();
			Graph *octree2 = new Graph;
			root = createNode(octree2, OctTreeNodeData(b2.centerx, b2.centery, b2.centerz));
			octree2->addNode(root);
			double radius = b2.diameter * 0.5;
			for (int i = 0; i < b2.nbodies; i++) {
				OctTreeNodeData &b = b2.body[i];
				b2.insert(octree2, root, b, radius);
			}
			b2.curr = 0;
			b2.computeCenterOfMass(octree2, root);

			for (int kk = 0; kk < b2.curr; kk++) {
				b2.computeForce(b2.leaf[kk], octree2, root,
						b2.diameter, b2.itolsq, step, b2.dthf,
						b2.epssq);
			}
			b2.advance(octree2, b2.dthf, b2.dtime);
			s_res = root.getData(Galois::Graph::NONE);
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
}
