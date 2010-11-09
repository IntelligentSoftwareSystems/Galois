/*
 * Node.h
 *
 *  Created on: Oct 18, 2010
 *      Author: amshali
 */

#ifndef NODE_H_
#define NODE_H_

#include <string>
#include <sstream>
#include <limits>

static const int DIST_INFINITY = std::numeric_limits<int>::max() / 2 - 1;

class SNode {
private:
public:
  int id;
  int dist;
  
 SNode(int _id) : id(_id), dist(DIST_INFINITY) {}
  std::string toString() {
    std::string ret;
    std::stringstream s(ret, std::ios_base::out);
    s << '[' << id << "] dist: " << dist;
    return ret;
  }
};

#endif /* NODE_H_ */
