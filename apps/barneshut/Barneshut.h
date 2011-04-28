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
#include <math.h>
#include <boost/math/constants/constants.hpp>

#include "OctTreeNodeData.h"
#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/IndexedGraph.h"
#include "Galois/Galois.h"
typedef Galois::Graph::IndexedGraph<OctTreeNodeData, int, true, 8> Graph;
typedef Galois::Graph::IndexedGraph<OctTreeNodeData, int, true, 8>::GraphNode GNode;

GNode createNode(Graph *octree, OctTreeNodeData b) {
	GNode newnode = octree->createNode(b);
	for (int i = 0; i < 8; i++) {
		octree->setNeighbor(newnode, GNode::GraphNode(), i, Galois::Graph::NONE);
	}
	return newnode;
}

class Barneshut {
public:
	double dtime; // length of one time step
	double eps; // potential softening parameter
	double tol; // tolerance for stopping recursion, <0.57 to bound error
	double dthf, epssq, itolsq;
	double diameter, centerx, centery, centerz;
  OctTreeNodeData* body; // the n bodies
	GNode* leaf;
  int seed;
	int curr;
	int nbodies; // number of bodies in system
	int ntimesteps; // number of time steps to run

	~Barneshut() {
		if (leaf != NULL) {
			delete[] leaf;
			leaf = NULL;
		}
		if (body != NULL) {
			delete[] body;
			body = NULL;
		}
	}

private:
  double nextDouble() {
    return rand() / (double) RAND_MAX;
  }

  void init(int _nbodies, int _ntimesteps, double _dtime, double _eps, double _tol, int _seed) {
    nbodies = _nbodies;
    ntimesteps = _ntimesteps;
    dtime = _dtime;
    eps = _eps;
    tol = _tol;
    dthf = 0.5 * dtime;
    epssq = eps * eps;
    itolsq = 1.0 / (tol * tol);
    seed = _seed;
    GaloisRuntime::getSystemThreadPool().getActiveThreads();
    body = new OctTreeNodeData[nbodies];
  }

	void insert(Graph *octree, GNode &root, OctTreeNodeData &b, double r) {
		double x = 0.0, y = 0.0, z = 0.0;
		OctTreeNodeData &n = root.getData(Galois::Graph::NONE);
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
		GNode child = octree->getNeighbor(root, i, Galois::Graph::NONE);
		if (child.isNull()) {
			GNode newnode = createNode(octree, b);
			octree->addNode(newnode, Galois::Graph::NONE);
			octree->setNeighbor(root, newnode, i, Galois::Graph::NONE);
		} else {
			double rh = 0.5 * r;
			OctTreeNodeData &ch = child.getData(Galois::Graph::NONE);
			if (!(ch.isLeaf())) {
				insert(octree, child, b, rh);
			} else {
				GNode newnode = createNode(octree, OctTreeNodeData(n.posx - rh + x,
						n.posy - rh + y, n.posz - rh + z));
				octree->addNode(newnode, Galois::Graph::NONE);
        assert(b.posx != n.posx && b.posy != n.posy && b.posz != n.posz);
				insert(octree, newnode, b, rh);
				insert(octree, newnode, ch, rh);
				octree->setNeighbor(root, newnode, i, Galois::Graph::NONE);
			}
		}
	}


public:
  void genInput(int nbodies, int ntimesteps, int _seed) {
    double r, v, x, y, z, sq, scale;
    double PI = boost::math::constants::pi<double>();

    init(nbodies, ntimesteps, 0.5, 0.05, 0.025, _seed);
    srand(seed);

    double rsc = (3 * PI) / 16;
    double vsc = sqrt(1.0 / rsc);

    for (int i = 0; i < nbodies; i++) {
      r = 1.0 / sqrt(pow(nextDouble() * 0.999, -2.0 / 3.0) - 1);
      do {
        x = nextDouble() * 2.0 - 1.0;
        y = nextDouble() * 2.0 - 1.0;
        z = nextDouble() * 2.0 - 1.0;
        sq = x * x + y * y + z * z;
      } while (sq > 1.0);
      scale = rsc * r / sqrt(sq);

      OctTreeNodeData &b = body[i];
      b.mass = 1.0 / nbodies;
      b.posx = x * scale;
      b.posy = y * scale;
      b.posz = z * scale;

      do {
        x = nextDouble();
        y = nextDouble() * 0.1;
      } while (y > x * x * pow(1 - x * x, 3.5));
      v = x * sqrt(2.0 / sqrt(1 + r * r));
      do {
        x = nextDouble() * 2.0 - 1.0;
        y = nextDouble() * 2.0 - 1.0;
        z = nextDouble() * 2.0 - 1.0;
        sq = x * x + y * y + z * z;
      } while (sq > 1.0);
      scale = vsc * v / sqrt(sq);
      b.setVelocity(x * scale, y * scale, z * scale);
    }
  }

	void readInput(const char *filename) {
		double vx, vy, vz;
		std::ifstream infile;
		infile.open(filename, std::ifstream::in);
		if (!infile) {
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

    init(nbodies, ntimesteps, tol, eps, dtime, 0);

		for (int i = 0; i < nbodies; i++) {
			OctTreeNodeData &b = body[i];
			infile >> b.mass;
			infile >> b.posx;
			infile >> b.posy;
			infile >> b.posz;
			infile >> vx;
			infile >> vy;
			infile >> vz;
			b.setVelocity(vx, vy, vz);
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
			OctTreeNodeData &b = body[i];
			posx = b.posx;
			posy = b.posy;
			posz = b.posz;
			if (minx > posx) 
				minx = posx;
			if (miny > posy) 
				miny = posy;
			if (minz > posz) 
				minz = posz;
			if (maxx < posx) 
				maxx = posx;
			if (maxy < posy) 
				maxy = posy;
			if (maxz < posz) 
				maxz = posz;
		}
		diameter = maxx - minx;
		if (diameter < (maxy - miny)) 
			diameter = (maxy - miny);
		if (diameter < (maxz - minz)) 
			diameter = (maxz - minz);
		centerx = (maxx + minx) * 0.5;
		centery = (maxy + miny) * 0.5;
		centerz = (maxz + minz) * 0.5;
	}

	void insertPoints(Graph *octree, GNode &root) {
		leaf = new GNode[nbodies];
		double radius = diameter * 0.5;
		for (int i = 0; i < nbodies; i++) {
			OctTreeNodeData &b = body[i];
			insert(octree, root, b, radius);
		}
  }

	void computeCenterOfMass(Graph *octree, GNode &root) {
		double m, px = 0.0, py = 0.0, pz = 0.0;
		OctTreeNodeData &n = root.getData(Galois::Graph::NONE);
		int j = 0;
		n.mass = 0.0;
		for (int i = 0; i < 8; i++) {
			GNode child = octree->getNeighbor(root, i, Galois::Graph::NONE);
			if (!child.isNull()) {
        // move non-null children to the front (needed later to make other code faster)
				if (i != j) {
					octree->setNeighbor(root, Graph::GraphNode::GraphNode(), i,
							Galois::Graph::NONE);
					octree->setNeighbor(root, child, j);
				}
				j++;
				OctTreeNodeData &ch = child.getData(Galois::Graph::NONE);
			  // sort bodies in tree order (approximation of
			  // putting nearby nodes together for locality)
      	if (ch.isLeaf()) {
					leaf[curr++] = child;
        } else {
					computeCenterOfMass(octree, child);
				}
				m = ch.mass;
				n.mass += m;
				px += ch.posx * m;
				py += ch.posy * m;
				pz += ch.posz * m;
			}
		}
		m = 1.0 / n.mass;
		n.posx = px * m;
		n.posy = py * m;
		n.posz = pz * m;
	}

	void computeForce(GNode &leaf, Graph *octree, GNode &root, double size,
			double itolsq, int step, double dthf, double epssq) {
		double ax, ay, az;
		OctTreeNodeData &nd = leaf.getData(Galois::Graph::NONE);

		ax = nd.accx;
		ay = nd.accy;
		az = nd.accz;
		nd.accx = 0.0;
		nd.accy = 0.0;
		nd.accz = 0.0;
		recurseForce(leaf, octree, root, size * size * itolsq, epssq);
		if (step > 0) {
			nd.velx += (nd.accx - ax) * dthf;
			nd.vely += (nd.accy - ay) * dthf;
			nd.velz += (nd.accz - az) * dthf;
		}
	}

	void recurseForce(GNode &leaf, Graph *octree, GNode &nn, double dsq,
			double epssq) {
		double drx, dry, drz, drsq, nphi, scale, idr;
		OctTreeNodeData &nd = leaf.getData(Galois::Graph::NONE);
		OctTreeNodeData &n = nn.getData(Galois::Graph::NONE);
		drx = n.posx - nd.posx;
		dry = n.posy - nd.posy;
		drz = n.posz - nd.posz;
		drsq = drx * drx + dry * dry + drz * drz;
		if (drsq < dsq) {
			if (!(n.isLeaf())) { // n is a cell
				dsq *= 0.25;
				GNode child = octree->getNeighbor(nn, 0, Galois::Graph::NONE);
				if (!child.isNull()) {
					recurseForce(leaf, octree, child, dsq, epssq);
					child = octree->getNeighbor(nn, 1, Galois::Graph::NONE);
					if (!child.isNull()) {
						recurseForce(leaf, octree, child, dsq, epssq);
						child = octree->getNeighbor(nn, 2, Galois::Graph::NONE);
						if (!child.isNull()) {
							recurseForce(leaf, octree, child, dsq, epssq);
							child = octree->getNeighbor(nn, 3, Galois::Graph::NONE);
							if (!child.isNull()) {
								recurseForce(leaf, octree, child, dsq, epssq);
								child = octree->getNeighbor(nn, 4, Galois::Graph::NONE);
								if (!child.isNull()) {
									recurseForce(leaf, octree, child, dsq, epssq);
									child = octree->getNeighbor(nn, 5, Galois::Graph::NONE);
									if (!child.isNull()) {
										recurseForce(leaf, octree, child, dsq, epssq);
										child = octree->getNeighbor(nn, 6, Galois::Graph::NONE);
										if (!child.isNull()) {
											recurseForce(leaf, octree, child, dsq, epssq);
											child = octree->getNeighbor(nn, 7, Galois::Graph::NONE);
											if (!child.isNull()) {
												recurseForce(leaf, octree, child, dsq, epssq);
											}
										}
									}
								}
							}
						}
					}
				}
			} else { // n is a body
				if (&n != &nd) {
					drsq += epssq;
					idr = 1 / sqrt(drsq);
					nphi = n.mass * idr;
					scale = nphi * idr * idr;
					nd.accx += drx * scale;
					nd.accy += dry * scale;
					nd.accz += drz * scale;
				}
			}
		} else { // node is far enough away, don't recurse any deeper
			drsq += epssq;
			idr = 1 / sqrt(drsq);
			nphi = n.mass * idr;
			scale = nphi * idr * idr;
			nd.accx += drx * scale;
			nd.accy += dry * scale;
			nd.accz += drz * scale;
		}
	}

	void advance(Graph *octree, double dthf, double dtime) {
		double dvelx, dvely, dvelz;
		double velhx, velhy, velhz;

		for (int i = 0; i < nbodies; i++) {
			OctTreeNodeData &nd = leaf[i].getData(Galois::Graph::NONE);
			dvelx = nd.accx * dthf;
			dvely = nd.accy * dthf;
			dvelz = nd.accz * dthf;
			velhx = nd.velx + dvelx;
			velhy = nd.vely + dvely;
			velhz = nd.velz + dvelz;
			nd.posx += velhx * dtime;
			nd.posy += velhy * dtime;
			nd.posz += velhz * dtime;
			nd.velx = velhx + dvelx;
			nd.vely = velhy + dvely;
			nd.velz = velhz + dvelz;
			body[i].restoreFrom(nd);
		}
	}
};

#endif /* BARNESHUT_H_ */
