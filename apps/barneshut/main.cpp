/*
 * main.cpp
 *
 *  Created on: Nov 11, 2010
 *      Author: amshali
 */

#include "Barneshut.h"
#include "Galois/Launcher.h"
#include "Galois/Runtime/Timer.h"
#include "Support/ThreadSafe/TSQueue.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <sys/time.h>

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

static const char* name = "Barnshut N-Body Simulator";
static const char* description = "Simulation of the gravitational forces in a galactic cluster using the Barnes-Hut n-body algorithm\n";
static const char* url = 0;
static const char* help = "<input file>";



Graph *octree;
GNode root;
int step;
Barneshut barneshut;

void process(GNode& item, Galois::Context<GNode>& lwl) {
	barneshut.computeForce(item, octree, root, barneshut.diameter,
			barneshut.itolsq, step, barneshut.dthf, barneshut.epssq);
}

int main(int argc, const char** argv) {
  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

  std::vector<const char*> args = parse_command_line(argc, argv, help);
  
  if (args.size() != 1) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

	barneshut.readInput(args[0], true);
	OctTreeNodeData res;
	std::cout<<"Num. of threads: "<<numThreads<<std::endl;
	Galois::setMaxThreads(numThreads);
	Galois::Launcher::startTiming();
	for (step = 0; step < barneshut.ntimesteps; step++) { // time-step the system
		barneshut.computeCenterAndDiameter();

    octree = new Graph;
		root = createNode(octree, OctTreeNodeData(barneshut.centerx,
				barneshut.centery, barneshut.centerz)); // create the
		// tree's
		// root
		octree->addNode(root);
		double radius = barneshut.diameter * 0.5;
		for (int i = 0; i < barneshut.nbodies; i++) {
			OctTreeNodeData &b = barneshut.body[i];
			barneshut.insert(octree, root, b, radius); // grow the tree by inserting
			// each body
		}
    barneshut.curr = 0;
		barneshut.computeCenterOfMass(octree, root); // summarize subtree info in each internal node (plus restructure tree and sort bodies for performance reasons)

		std::vector<GNode> wl;
		for (int ii = 0; ii < barneshut.curr; ii++) {
			wl.push_back(barneshut.leaf[ii]);
		}
		Galois::for_each(wl.begin(), wl.end(), process);

		barneshut.advance(octree, barneshut.dthf, barneshut.dtime); // advance the position and velocity of each

		if (Galois::Launcher::isFirstRun()) {
			// print center of mass for this timestep
			res = root.getData(Galois::Graph::NONE);
			std::cout << "Timestep " << step << " Center of Mass = " << res.posx
					<< " " << res.posy << " " << res.posz << std::endl;
		}
		delete octree;
	} // end of time step
	Galois::Launcher::stopTiming();
	std::cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";

	if (Galois::Launcher::isFirstRun()) { // verify result
		Barneshut barneshut2;
		barneshut2.readInput(args[0], false);
		OctTreeNodeData s_res;
		for (step = 0; step < barneshut2.ntimesteps; step++) {
			barneshut2.computeCenterAndDiameter();
			Graph *octree2 = new Graph;
			root = createNode(octree2, OctTreeNodeData(barneshut2.centerx,
					barneshut2.centery, barneshut2.centerz)); // create the
			octree2->addNode(root);
			double radius = barneshut2.diameter * 0.5;
			for (int i = 0; i < barneshut2.nbodies; i++) {
				OctTreeNodeData &b = barneshut2.body[i];
				barneshut2.insert(octree2, root, b, radius);
			}
			barneshut2.curr = 0;
			barneshut2.computeCenterOfMass(octree2, root);

			for (int kk = 0; kk < barneshut2.curr; kk++) {
				barneshut2.computeForce(barneshut2.leaf[kk], octree2, root,
						barneshut2.diameter, barneshut2.itolsq, step, barneshut2.dthf,
						barneshut2.epssq);
			}
			barneshut2.advance(octree2, barneshut2.dthf, barneshut2.dtime);
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
