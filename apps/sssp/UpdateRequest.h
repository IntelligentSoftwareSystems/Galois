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
  GNode n;
  int w;
 UpdateRequest() :n(), w(0) {};
  UpdateRequest(GNode _n, int _w)
    :n(_n), w(_w)
  {}
  
  bool operator> (const UpdateRequest& RHS) const {
    return w > RHS.w;
  }

  bool operator<(const UpdateRequest& RHS) const {
    return w < RHS.w;
  }
};

#endif /* UPDATEREQUEST_H_ */
