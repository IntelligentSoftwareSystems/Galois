/*
 * OctTreeNodeData.h
 *
 *  Created on: Nov 11, 2010
 *      Author: amshali
 */

#ifndef OCTTREENODEDATA_H_
#define OCTTREENODEDATA_H_
#include <string>
#include <sstream>
#include <limits>

class OctTreeNodeData {
public:
	double mass;
	double posx;
	double posy;
	double posz;
	double velx;
	double vely;
	double velz;
	double accx;
	double accy;
	double accz;
	bool leaf;
	OctTreeNodeData() {
		leaf = true;
		mass = 0.0;
		posx = 0.0;
		posy = 0.0;
		posz = 0.0;
		velx = 0.0;
		vely = 0.0;
		velz = 0.0;
		accx = 0.0;
		accy = 0.0;
		accz = 0.0;
	}
	OctTreeNodeData(double px, double py, double pz) {
		mass = 0.0;
		posx = px;
		posy = py;
		posz = pz;
		velx = 0.0;
		vely = 0.0;
		velz = 0.0;
		accx = 0.0;
		accy = 0.0;
		accz = 0.0;
		leaf = false;
	}
	OctTreeNodeData(const OctTreeNodeData& copy) {
		restoreFrom(copy);
	}
	bool isLeaf() const {
		return leaf;
	}
	void setVelocity(double x, double y, double z) {
		velx = x;
		vely = y;
		velz = z;
	}
	std::string toString() const {
		std::ostringstream s;
		if (isLeaf()) {
			s << "pos = (" << posx << "," << posy << "," << posz << ") ";
			s << "vel = (" << velx << "," << vely << "," << velz << ") ";
			s << "acc = (" << accx << "," << accy << "," << accz << ") ";
                        s << "mass = " << mass;
			return s.str();
		} else {
			s << "mass = " << mass << " pos = (" << posx << "," << posy << ","
					<< posz << ")";
		}
		return s.str();
	}
	double _posx() {
		return posx;
	}
	double _posy() {
		return posy;
	}
	double _posz() {
		return posz;
	}
	void restoreFrom(const OctTreeNodeData& data) {
		posx = data.posx;
		posy = data.posy;
		posz = data.posz;
		mass = data.mass;
		posx = data.posx;
		posy = data.posy;
		posz = data.posz;
		mass = data.mass;
		velx = data.velx;
		vely = data.vely;
		velz = data.velz;
		accx = data.accx;
		accy = data.accy;
		accz = data.accz;
		leaf = data.leaf;
	}
};

std::ostream& operator<<(std::ostream& os, const OctTreeNodeData& b) {
  os << b.toString();
  return os;
}

#endif /* OCTTREENODEDATA_H_ */
