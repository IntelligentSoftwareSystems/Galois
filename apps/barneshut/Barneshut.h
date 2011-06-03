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
#include "Galois/Galois.h"
#include "Galois/Graphs/Graph.h"

struct OctreeInternal {
  OctreeInternal* child[8];
  OctTreeNodeData data;
  OctreeInternal(OctTreeNodeData _data) : data(_data) { bzero(child, sizeof(*child) * 8); }
  OctTreeNodeData& getData(int) {
    return data;
  }
  OctreeInternal* getNeighbor(OctreeInternal* node, int i, int) {
    return node->child[i];
  }
  void addNode(OctreeInternal*, int) {

  }
  void addNode(OctreeInternal*) {

  }
  void setNeighbor(OctreeInternal* node, OctreeInternal* value, int i, int) {
    node->child[i] = value;
  }
  virtual ~OctreeInternal() {
    for (int i = 0; i < 8; i++) {
      if (child[i] != NULL) {
        delete child[i];
      }
    }
  }
};

static OctreeInternal* createNode(OctreeInternal* , OctTreeNodeData b) {
  OctreeInternal* n = new OctreeInternal(b);
	return n;
}

typedef OctreeInternal Graph;
typedef Graph* GNode;

class Barneshut {
public:
	double dtime; // length of one time step
	double eps; // potential softening parameter
	double tol; // tolerance for stopping recursion, <0.57 to bound error
	double dthf, epssq, itolsq;
	double diameter, centerx, centery, centerz;
  OctTreeNodeData* body; // the n bodies
	OctTreeNodeData* leaf;
  std::vector<OctTreeNodeData> *partitions;
  int seed;
	int curr;
	int nbodies; // number of bodies in system
	int ntimesteps; // number of time steps to run

  Barneshut() : body(NULL), leaf(NULL), partitions(NULL) { }

	~Barneshut() {
		if (leaf != NULL) {
			delete[] leaf;
			leaf = NULL;
		}
		if (body != NULL) {
			delete[] body;
			body = NULL;
		}
    if (partitions != NULL) {
      delete[] partitions;
      partitions = NULL;
    }
	}

private:
  double next_double() {
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
    body = new OctTreeNodeData[nbodies];
  }

	void insert(Graph *octree, GNode &root, OctTreeNodeData &b, double r) {
		double x = 0.0, y = 0.0, z = 0.0;
		OctTreeNodeData &n = root->getData(Galois::Graph::NONE);
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
		//if (child.isNull()) {
		if (child == NULL) {
			GNode newnode = createNode(octree, b);
			octree->addNode(newnode, Galois::Graph::NONE);
			octree->setNeighbor(root, newnode, i, Galois::Graph::NONE);
		} else {
			double rh = 0.5 * r;
			OctTreeNodeData &ch = child->getData(Galois::Graph::NONE);
			if (!(ch.isLeaf())) {
				insert(octree, child, b, rh);
			} else {
				GNode newnode = createNode(octree, OctTreeNodeData(n.posx - rh + x,
						n.posy - rh + y, n.posz - rh + z));
				octree->addNode(newnode, Galois::Graph::NONE);
				insert(octree, newnode, b, rh);
				insert(octree, newnode, ch, rh);
				octree->setNeighbor(root, newnode, i, Galois::Graph::NONE);
			}
		}
	}

  void generate_uniform_input(int nbodies, int ntimesteps, int _seed) {
    double scale = 0.01;
    init(nbodies, ntimesteps, 0.5, 0.05, 0.025, _seed);
    srand(seed);

    for (int i = 0; i < nbodies; i++) {
      OctTreeNodeData &b = body[i];
      b.mass = 1.0 / nbodies;
      b.posx = next_double();
      b.posy = next_double();
      b.posz = next_double();
      double velx = next_double() * scale;
      double vely = next_double() * scale;
      double velz = next_double() * scale;
      b.setVelocity(velx, vely, velz);
    }
  }
  
  void generate_plummer_input(int nbodies, int ntimesteps, int _seed) {
    double r, v, x, y, z, sq, scale;
    double PI = boost::math::constants::pi<double>();

    init(nbodies, ntimesteps, 0.5, 0.05, 0.025, _seed);
    srand(seed);

    double rsc = (3 * PI) / 16;
    double vsc = sqrt(1.0 / rsc);

    for (int i = 0; i < nbodies; i++) {
      r = 1.0 / sqrt(pow(next_double() * 0.999, -2.0 / 3.0) - 1);
      do {
        x = next_double() * 2.0 - 1.0;
        y = next_double() * 2.0 - 1.0;
        z = next_double() * 2.0 - 1.0;
        sq = x * x + y * y + z * z;
      } while (sq > 1.0);
      scale = rsc * r / sqrt(sq);

      OctTreeNodeData &b = body[i];
      b.mass = 1.0 / nbodies;
      b.posx = x * scale;
      b.posy = y * scale;
      b.posz = z * scale;

      do {
        x = next_double();
        y = next_double() * 0.1;
      } while (y > x * x * pow(1 - x * x, 3.5));
      v = x * sqrt(2.0 / sqrt(1 + r * r));
      do {
        x = next_double() * 2.0 - 1.0;
        y = next_double() * 2.0 - 1.0;
        z = next_double() * 2.0 - 1.0;
        sq = x * x + y * y + z * z;
      } while (sq > 1.0);
      scale = vsc * v / sqrt(sq);
      b.setVelocity(x * scale, y * scale, z * scale);
    }
  }

public:
  void genInput(int nbodies, int ntimesteps, int _seed) {
    generate_plummer_input(nbodies, ntimesteps, _seed);
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
		leaf = new OctTreeNodeData[nbodies];
		double radius = diameter * 0.5;
		for (int i = 0; i < nbodies; i++) {
			OctTreeNodeData &b = body[i];
			insert(octree, root, b, radius);
		}
  }

	void computeCenterOfMass(Graph *octree, GNode &root) {
		double m, px = 0.0, py = 0.0, pz = 0.0;
		OctTreeNodeData &n = root->getData(Galois::Graph::NONE);
		int j = 0;
		n.mass = 0.0;
		for (int i = 0; i < 8; i++) {
			GNode child = octree->getNeighbor(root, i, Galois::Graph::NONE);
			//if (!child.isNull()) {
			if (child != NULL) {
        // move non-null children to the front (needed later to make other code faster)
				if (i != j) {
					octree->setNeighbor(root, NULL, i, Galois::Graph::NONE);
					octree->setNeighbor(root, child, j, Galois::Graph::NONE);
				}
				j++;
				OctTreeNodeData &ch = child->getData(Galois::Graph::NONE);
			  // sort bodies in tree order (approximation of
			  // putting nearby nodes together for locality)
      	if (ch.isLeaf()) {
					leaf[curr++] = ch;
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

	void computeForce(OctTreeNodeData &nd, Graph *octree, GNode &root, double size,
			double itolsq, int step, double dthf, double epssq) {
		double ax, ay, az;

		ax = nd.accx;
		ay = nd.accy;
		az = nd.accz;
		nd.accx = 0.0;
		nd.accy = 0.0;
		nd.accz = 0.0;
		//recurseForce(nd, octree, root, size * size * itolsq, epssq);
		iterateForce(nd, octree, root, size * size * itolsq, epssq);
		if (step > 0) {
			nd.velx += (nd.accx - ax) * dthf;
			nd.vely += (nd.accy - ay) * dthf;
			nd.velz += (nd.accz - az) * dthf;
		}
	}

  void partition(Graph* root, int levels) {
    int numPartitions = pow(8, levels);
    assert(partitions == NULL);
    partitions = new std::vector<OctTreeNodeData>[numPartitions];

		for (int i = 0; i < nbodies; i++) {
			OctTreeNodeData &b = body[i];
      int l = 0;
      int acc = 0;
      OctreeInternal* node = root;
      while (node != NULL && l <= levels) {
        int index = 0;
        if (node->data.posx < b.posx) {
          index = 1;
        }
        if (node->data.posy < b.posy) {
          index += 2;
        }
        if (node->data.posz < b.posz) {
          index += 4;
        }
        l++;
        acc = (acc << 3) + index;
        node = static_cast<OctreeInternal*>(node->child[index]);
      } 
      // acc is index of NULL cell, correct for that
      acc = acc >> 3;
      partitions[acc].push_back(b);
    }
  }

  struct Frame {
    double dsq;
    GNode node;
    Frame(GNode _node, double _dsq): dsq(_dsq), node(_node) { }
  };

  void iterateForce(OctTreeNodeData &nd, Graph *octree, GNode &root, double root_dsq, double epssq) {
    std::vector<Frame> stack;
    stack.push_back(Frame(root, root_dsq));

    while (!stack.empty()) {
      Frame f = stack.back();
      stack.pop_back();

		  OctTreeNodeData &n = f.node->getData(Galois::Graph::NONE);
      double drx = n.posx - nd.posx;
      double dry = n.posy - nd.posy;
      double drz = n.posz - nd.posz;
      double drsq = drx * drx + dry * dry + drz * drz;
      if (drsq < f.dsq) {
        if (!(n.isLeaf())) { // n is a cell
          double dsq = f.dsq * 0.25;
          for (int i = 0; i < 8; i++) {
            GNode child = octree->getNeighbor(f.node, i, Galois::Graph::NONE);
            if (child != NULL) {
              stack.push_back(Frame(child, dsq));
            } else {
              break;
            }
          }
        } else { // n is a body
          if (&n != &nd) {
            drsq += epssq;
            double idr = 1 / sqrt(drsq);
            double nphi = n.mass * idr;
            double scale = nphi * idr * idr;
            nd.accx += drx * scale;
            nd.accy += dry * scale;
            nd.accz += drz * scale;
          } else {

          }
        }
      } else { // node is far enough away, don't recurse any deeper
        drsq += epssq;
        double idr = 1 / sqrt(drsq);
        double nphi = n.mass * idr;
        double scale = nphi * idr * idr;
        nd.accx += drx * scale;
        nd.accy += dry * scale;
        nd.accz += drz * scale;
      }
    }
  }

	void recurseForce(OctTreeNodeData &nd, Graph *octree, GNode &nn, double dsq,
			double epssq) {
		double drx, dry, drz, drsq, nphi, scale, idr;
		OctTreeNodeData &n = nn->getData(Galois::Graph::NONE);
		drx = n.posx - nd.posx;
		dry = n.posy - nd.posy;
		drz = n.posz - nd.posz;
		drsq = drx * drx + dry * dry + drz * drz;
		if (drsq < dsq) {
			if (!(n.isLeaf())) { // n is a cell
				dsq *= 0.25;
        for (int i = 0; i < 8; i++) {
          GNode child = octree->getNeighbor(nn, i, Galois::Graph::NONE);
          if (child != NULL) {
            recurseForce(nd, octree, child, dsq, epssq);
          } else {
            break;
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
        } else {

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
			OctTreeNodeData &nd = leaf[i];
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
		}
	}
};

#endif /* BARNESHUT_H_ */
