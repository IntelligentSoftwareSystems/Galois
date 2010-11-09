/*
 * UpdateRequest.h
 *
 *  Created on: Oct 26, 2010
 *      Author: amshali
 */

#ifndef UPDATEREQUEST_H_
#define UPDATEREQUEST_H_

class UpdateRequest {
private:
public:
  bool light;
  GNode n;
  int w;
  UpdateRequest() {};
  UpdateRequest(GNode _n, int _w, bool _light)
    :light(_light), n(_n), w(_w)
  {}
};

#endif /* UPDATEREQUEST_H_ */
