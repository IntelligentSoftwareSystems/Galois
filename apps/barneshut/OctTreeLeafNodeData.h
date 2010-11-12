/*
 * OctTreeLeafNodeData.h
 *
 *  Created on: Nov 11, 2010
 *      Author: amshali
 */

#ifndef OCTTREELEAFNODEDATA_H_
#define OCTTREELEAFNODEDATA_H_

#include "OctTreeNodeData.h"

class OctTreeLeafNodeData: public OctTreeNodeData {
public:
	double velx;
	double vely;
	double velz;
	double accx;
	double accy;
	double accz;
	OctTreeLeafNodeData() :
		OctTreeNodeData(0.0, 0.0, 0.0) {
		velx = 0.0;
		vely = 0.0;
		velz = 0.0;
		accx = 0.0;
		accy = 0.0;
		accz = 0.0;
	}

	OctTreeLeafNodeData(const OctTreeLeafNodeData& copy) :
		OctTreeNodeData(copy.posx, copy.posy, copy.posz) {
		mass = copy.mass;
		velx = copy.velx;
		vely = copy.vely;
		velz = copy.velz;
		accx = copy.accx;
		accy = copy.accy;
		accz = copy.accz;
	}

	bool isLeaf() {
		return true;
	}
	void setVelocity(double x, double y, double z) {
		velx = x;
		vely = y;
		velz = z;
	}
	bool equals(const OctTreeLeafNodeData& n) {
		return (this->posx == n.posx && this->posy == n.posy && this->posz
				== n.posz);
	}
	std::string toString() {
		std::ostringstream s;
		s << OctTreeNodeData::toString();
		s << "vel = (" << velx << "," << vely << "," << velz << ")";
		s << "acc = (" << accx << "," << accy << "," << accz << ")";
		return s.str();
	}
	void restoreFrom(OctTreeLeafNodeData data) {
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
	}
};

#endif /* OCTTREELEAFNODEDATA_H_ */
