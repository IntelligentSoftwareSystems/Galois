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
  int idx;
  UpdateRequest() {};
  UpdateRequest(GNode _n, int _w, bool _light)
    :light(_light), n(_n), w(_w)
  { idx = computeIndx(); }

  int computeIndx() const {
    const int range = 30*1024;
    int bucket1 = std::min(w / 700, range - 1);
    int retval1;
    if (light)
      retval1 = bucket1 * 2;
    else
      retval1 = bucket1 * 2 + 1;
    return retval1;
  }
  
  bool operator> (const UpdateRequest& RHS) const {
    return idx > RHS.idx; //computeIndx() > RHS.computeIndx();
  }

  bool operator<(const UpdateRequest& RHS) const {
    return idx < RHS.idx; //computeIndx() < RHS.computeIndx();
  }
};

#endif /* UPDATEREQUEST_H_ */
