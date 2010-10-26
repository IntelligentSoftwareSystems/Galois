/*
 * SSSP.cpp
 *
 *  Created on: Oct 18, 2010
 *      Author: reza
 */

#include <list>
#include "SSSP.h"

SSSP::SSSP() {
}

SSSP::~SSSP() {
}

void SSSP::bellman_ford(const std::list<SNode> & nodes,
		const std::list<SEdge> & edges, SNode & source) {

	source.set_dist(0);
	source.set_pred(NULL);

	for (unsigned int i = 0; i < nodes.size() - 1; i++) {
		std::list<SEdge>::const_iterator it;
		for (it = edges.begin(); it != edges.end(); it++) {
			SNode & u = (*it).get_source();
			SNode & v = (*it).get_destination();
			if (u.get_dist() + (*it).get_weight() < v.get_dist()) {
				v.set_dist(u.get_dist() + (*it).get_weight());
				v.set_pred(&u);
			}
		}
	}

}

int main(int argc, char** argv) {
  return 0;
}

