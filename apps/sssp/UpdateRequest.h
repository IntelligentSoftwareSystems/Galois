/*
 * UpdateRequest.h
 *
 *  Created on: Oct 26, 2010
 *      Author: amshali
 */

#ifndef UPDATEREQUEST_H_
#define UPDATEREQUEST_H_

#include "SSSP.h"

class UpdateRequest {
private:
public:
	bool light;
	GNode n;
	int w;
	UpdateRequest() {};
	UpdateRequest(GNode _n, int _w, bool _light) {
		w = _w;
		n = _n;
		light = _light;
	}
	;
	virtual ~UpdateRequest() {};
};

#endif /* UPDATEREQUEST_H_ */
