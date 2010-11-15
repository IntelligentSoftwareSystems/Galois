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

Graph *octree;
GNode root;
int step;
Barneshut barneshut;

void process(GNode& item, Galois::Context<GNode>& lwl) {
	barneshut.computeForce(item, octree, root, barneshut.diameter,
			barneshut.itolsq, step, barneshut.dthf, barneshut.epssq);
}

int main(int argc, char* argv[]) {
  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

	if (Galois::Launcher::isFirstRun()) {
		std::cerr << "Lonestar Benchmark Suite v3.0" << std::endl;
		std::cerr
				<< "Copyright (C) 2007, 2008, 2009, 2010 The University of Texas at Austin"
				<< std::endl;
		std::cerr << "http://iss.ices.utexas.edu/lonestar/" << std::endl;
		std::cerr << std::endl;
		std::cerr << "Simulation of the gravitational forces in a galactic"
				<< std::endl;
		std::cerr << "cluster using the Barnes-Hut n-body algorithm" << std::endl;
		std::cerr << "http://iss.ices.utexas.edu/lonestar/barneshut.html"
				<< std::endl;
		std::cerr << std::endl;
	}
	if (argc != 3) {
		std::cerr << "arguments: <num_threads> <input_file_name>" << std::endl;
		exit(-1);
	}

	int threadnum = atoi(argv[1]);
	barneshut.readInput(argv[2], true);
	OctTreeNodeData res;
	Galois::setMaxThreads(threadnum);

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
			barneshut.insert(octree, root, *barneshut.body[i], radius); // grow the tree by inserting
			// each body
		}
		barneshut.curr = 0;
		barneshut.computeCenterOfMass(octree, root); // summarize subtree info in each internal node (plus restructure tree and sort bodies for performance reasons)

		threadsafe::ts_queue<GNode> wl;
		for (int ii = 0; ii < barneshut.curr; ii++) {
			wl.push(barneshut.leaf[ii]);
		}
		Galois::for_each(wl, process);

		barneshut.advance(octree, barneshut.dthf, barneshut.dtime); // advance the position and velocity of each

		if (Galois::Launcher::isFirstRun()) {
			// print center of mass for this timestep
			res = root.getData();
			std::cout << "Timestep " << step << " Center of Mass = " << res.posx
					<< " " << res.posy << " " << res.posz << std::endl;
		}
//		delete octree;
	} // end of time step
	Galois::Launcher::stopTiming();
	std::cout << "STAT: Time " << Galois::Launcher::elapsedTime() << "\n";
//	barneshut.clear();

	if (Galois::Launcher::isFirstRun()) { // verify result
		Barneshut barneshut2;
		barneshut2.readInput(argv[2], false);
		OctTreeNodeData s_res;

		for (step = 0; step < barneshut2.ntimesteps; step++) {
			barneshut2.computeCenterAndDiameter();
			Graph *octree2 = new Graph;
			root = createNode(octree2, OctTreeNodeData(barneshut2.centerx,
					barneshut2.centery, barneshut2.centerz)); // create the
			octree2->addNode(root);
			double radius = barneshut2.diameter * 0.5;
			for (int i = 0; i < barneshut2.nbodies; i++) {
				barneshut2.insert(octree2, root, *barneshut2.body[i], radius);
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
//			delete octree2;
		}

		if ((fabs(res.posx - s_res.posx) / fabs(std::min(res.posx, s_res.posx))
				> 0.001) || (fabs(res.posy - s_res.posy) / fabs(std::min(res.posy,
				s_res.posy)) > 0.001) || (fabs(res.posz - s_res.posz) / fabs(std::min(
				res.posz, s_res.posz)) > 0.001)) {
			std::cerr << "verification failed" << std::endl;
		} else {
			std::cerr << "verification succeeded" << std::endl;
		}
//		barneshut2.clear();
	}
}
