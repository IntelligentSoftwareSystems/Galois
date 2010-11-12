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
	OctTreeNodeData(double px, double py, double pz) {
		mass = 0.0;
		posx = px;
		posy = py;
		posz = pz;
	}
	OctTreeNodeData(const OctTreeNodeData& copy) {
		mass = copy.mass;
		posx = copy.posx;
		posy = copy.posy;
		posz = copy.posz;
	}
	virtual bool isLeaf() {
		return false;
	}
	std::string toString() {
		std::ostringstream s;
		s << "mass = " << mass << " pos = (" << posx << "," << posy << "," << posz
				<< ")";
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
	}
};

#endif /* OCTTREENODEDATA_H_ */
