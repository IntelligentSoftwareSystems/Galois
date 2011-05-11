/*
 * Verifier.h
 *
 *  Created on: May 11, 2011
 *      Author: xinsui
 */

#ifndef VERIFIER_H_
#define VERIFIER_H_
#include "Element.h"
#include "Tuple.h"
#include <stack>
#include <set>
using namespace std;
class Verifier {
public:
	Verifier(){}
	virtual ~Verifier(){}

	bool checkConsistency(Graph* graph){
		bool error = false;
		for (Graph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;

			DTElement& element = node.getData(Galois::Graph::NONE, 0);

			if (element.getDim() == 2) {
				if (graph->neighborsSize(node, Galois::Graph::NONE, 0) != 1) {
					std::cerr << "-> Segment " << element << " has " << graph->neighborsSize(node, Galois::Graph::NONE, 0) << " relation(s)\n";
					error = true;
				}
			} else if (element.getDim() == 3) {
				if (graph->neighborsSize(node, Galois::Graph::NONE, 0) != 3) {
					std::cerr << "-> Triangle " << element << " has " << graph->neighborsSize(node, Galois::Graph::NONE, 0) << " relation(s)";
					error = true;
				}
			} else {
				std::cerr << "-> Figures with " << element.getDim() << " edges";
				error = true;
			}
		}
		if (error)
			return false;
		return true;
	}

	bool checkReachability(Graph* graph){
		std::stack<GNode> remaining;
		std::set<GNode> found;
		remaining.push(*(graph->active_begin()));

		while (!remaining.empty()) {
			GNode node = remaining.top();
			remaining.pop();
			if (!found.count(node)) {
				assert(graph->containsNode(node) && "Reachable node was removed from graph");
				found.insert(node);
				int i = 0;
				for (Graph::neighbor_iterator ii = graph->neighbor_begin(node, Galois::Graph::NONE, 0), ee = graph->neighbor_end(node, Galois::Graph::NONE, 0); ii != ee; ++ii) {
					assert(i < 3);
					assert(graph->containsNode(*ii));
					assert(node != *ii);
					++i;
					//	  if (!found.count(*ii))
					remaining.push(*ii);
				}
			}
		}

		if (found.size() != graph->size()) {
			std::cerr << "Not all elements are reachable \n";
			std::cerr << "Found: " << found.size() << "\nMesh: " << graph->size() << "\n";
			assert(0 && "Not all elements are reachable");
			return false;
		}
		return true;
	}

	bool checkDelaunayProperty(Graph* graph){
		for (Graph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			DTElement& e = node.getData(Galois::Graph::NONE, 0);
			for (Graph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE, 0), eejj = graph->neighbor_end(node, Galois::Graph::NONE, 0); jj != eejj; ++jj) {
				GNode neighborNode = *jj;
				DTElement& e2 = neighborNode.getData(Galois::Graph::NONE, 0);
				if (e.getDim() == 3 && e2.getDim() == 3) {
					const DTTuple* t2 = getTupleT2OfRelatedEdge(e, e2);
					if (!t2) {
						std::cerr << "missing tuple \n";
						return false;
					}
					if (e.inCircle(*t2)) {
						std::cerr << "violate delaunay property \n";
						return false;
					}
				}
			}
		}
		return true;
	}

private:
	const DTTuple* getTupleT2OfRelatedEdge(DTElement& e1, DTElement& e2) {
		int e2_0 = -1;
		int e2_1 = -1;
		int phase = 0;

		for (int i = 0; i < e1.getDim(); i++) {
			for (int j = 0; j < e2.getDim(); j++) {
				if (e1.getPoint(i) == e2.getPoint(j)) {
					if (phase == 0) {
						e2_0 = j;
						phase = 1;
						break;
					} else {
						e2_1 = j;
						for (int k = 0; k < 3; k++) {
							if (k != e2_0 && k != e2_1) {
								 return &(e2.getPoint(k));
							}
						}
					}
				}
			}
		}
		return NULL;
	}
};

#endif /* VERIFIER_H_ */
