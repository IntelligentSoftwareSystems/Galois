/*
 * Barneshut.h
 *
 *  Created on: Nov 11, 2010
 *      Author: amshali
 */

#ifndef BARNESHUT_H_
#define BARNESHUT_H_
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>

#include "OctTreeNodeData.h"
#include "OctTreeLeafNodeData.h"
#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/IndexedGraph.h"
#include "Galois/Galois.h"
typedef Galois::Graph::IndexedGraph<OctTreeNodeData, int, true, 8> Graph;
typedef Galois::Graph::IndexedGraph<OctTreeNodeData, int, true, 8>::GraphNode GNode;

class Barneshut {
public:
	int nbodies; // number of bodies in system
	int ntimesteps; // number of time steps to run
	double dtime; // length of one time step
	double eps; // potential softening parameter
	double tol; // tolerance for stopping recursion, should be less than 0.57 for 3D case to bound error

	double dthf, epssq, itolsq;
	OctTreeLeafNodeData *body; // the n bodies
	GNode *leaf;
	double diameter, centerx, centery, centerz;
	int curr;

	void readInput(char *filename, bool print) {
		double vx, vy, vz;
		std::ifstream infile;
		infile.open(filename, std::ifstream::in); // opens the vector file
		if (!infile) { // file couldn't be opened
			std::cerr << "Error: vector file could not be opened" << std::endl;
			exit(-1);
		}

		std::string line;
		infile >> nbodies;
		getline(infile, line);
		infile >> ntimesteps;
		getline(infile, line);
		infile >> dtime;
		getline(infile, line);
		infile >> eps;
		getline(infile, line);
		infile >> tol;
		getline(infile, line);
		dthf = 0.5 * dtime;
		epssq = eps * eps;
		itolsq = 1.0 / (tol * tol);
		if (print) {
			std::cerr << "configuration: " << nbodies << " bodies, " << ntimesteps
					<< " time steps" << std::endl << std::endl;
		}
		body = new OctTreeLeafNodeData[nbodies];
		leaf = new GNode[nbodies];
		for (int i = 0; i < nbodies; i++) {
			body[i] = OctTreeLeafNodeData();
		}
		for (int i = 0; i < nbodies; i++) {
			infile >> body[i].mass;
			infile >> body[i].posx;
			infile >> body[i].posy;
			infile >> body[i].posz;
			infile >> vx;
			infile >> vy;
			infile >> vz;
			body[i].setVelocity(vx, vy, vz);
			getline(infile, line);
		}
	}

	void computeCenterAndDiameter() {
		double minx, miny, minz;
		double maxx, maxy, maxz;
		double posx, posy, posz;
		minx = miny = minz = std::numeric_limits<double>::max();
		maxx = maxy = maxz = std::numeric_limits<double>::min();
		for (int i = 0; i < nbodies; i++) {
			posx = body[i].posx;
			posy = body[i].posy;
			posz = body[i].posz;
			if (minx > posx) {
				minx = posx;
			}
			if (miny > posy) {
				miny = posy;
			}
			if (minz > posz) {
				minz = posz;
			}
			if (maxx < posx) {
				maxx = posx;
			}
			if (maxy < posy) {
				maxy = posy;
			}
			if (maxz < posz) {
				maxz = posz;
			}
		}
		diameter = maxx - minx;
		if (diameter < (maxy - miny)) {
			diameter = (maxy - miny);
		}
		if (diameter < (maxz - minz)) {
			diameter = (maxz - minz);
		}
		centerx = (maxx + minx) * 0.5;
		centery = (maxy + miny) * 0.5;
		centerz = (maxz + minz) * 0.5;
	}
	void insert(Graph *octree, GNode &root, OctTreeNodeData &b, double r) {
		double x = 0.0, y = 0.0, z = 0.0;
		OctTreeNodeData n = root.getData();
		int i = 0;
		if (n.posx < b.posx) {
			i = 1;
			x = r;
		}
		if (n.posy < b.posy) {
			i += 2;
			y = r;
		}
		if (n.posz < b.posz) {
			i += 4;
			z = r;
		}
		GNode child = octree->getNeighbor(root, i,Galois::Graph::NONE);
		if (child.isIDNull()) {
			GNode newnode = octree->createNode(b);
			octree->addNode(newnode, Galois::Graph::NONE);
			octree->setNeighbor(root, newnode, i, Galois::Graph::NONE);
		} else {
			double rh = 0.5 * r;
			OctTreeNodeData ch = child.getData();
			if (!(ch.isLeaf())) {
				insert(octree, child, b, rh);
			} else {
				GNode newnode = octree->createNode(OctTreeNodeData(
						n.posx - rh + x, n.posy - rh + y, n.posz - rh + z));
				octree->addNode(newnode, Galois::Graph::NONE);
				insert(octree, newnode, b, rh);
				insert(octree, newnode, ch, rh);
				octree->setNeighbor(root, newnode, i, Galois::Graph::NONE);
			}
		}
	}
};

#endif /* BARNESHUT_H_ */
