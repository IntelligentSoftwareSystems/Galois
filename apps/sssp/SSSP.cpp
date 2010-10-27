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

void SSSP::updateSourceAndSink(const int sourceId, const int sinkId) {
	for (Graph::active_iterator src = graph->active_begin(), ee =
			graph->active_end(); src != ee; ++src) {
		SNode& node = src->getData();
		node.set_dist(INFINITY);
		if (node.id == sourceId) {
			source = *src;
			node.set_dist(0);
		} else if (node.id == sinkId) {
			sink = *src;
		}
	}
}

int SSSP::getEdgeData(GNode src, GNode dst) {
	if (executorType.bfs)
		return 1;
	else
		return graph->getEdgeData(src, dst).get_weight();
}

void SSSP::verify() {
	if (source.getData().get_dist() != 0) {
		cerr << "source has non-zero dist value" << endl;
		exit(-1);
	}

	for (Graph::active_iterator src = graph->active_begin(), ee =
			graph->active_end(); src != ee; ++src) {
		const int dist = src->getData().get_dist();
		if (dist >= INFINITY) {
			cerr << "found node = " << src->getData().get_dist()
					<< " with label >= INFINITY = " << dist << endl;
			exit(-1);
		}

		for (Graph::neighbor_iterator ii = graph->neighbor_begin(*src), ee =
				graph->neighbor_end(*src); ii != ee; ++ii) {
			GNode neighbor = *ii;
			int ddist = src->getData().get_dist();

			if (ddist > dist + getEdgeData(neighbor, *src)) { // FIXME:might be wrong
				cerr << "bad level value at " << src->getData().id
						<< " which is a neighbor of " << neighbor.getData().id << endl;
				exit(-1);
			}

		}
	}

	cerr << "result verified" << endl;
}

void SSSP::runBody(const GNode src) {
	vector<UpdateRequest> initial;
	for (Graph::neighbor_iterator ii = graph->neighbor_begin(src), ee =
			graph->neighbor_end(src); ii != ee; ++ii) {
		GNode dst = *ii;
		int w = getEdgeData(src, dst);
		UpdateRequest up = UpdateRequest(dst, w, w <= delta);
		initial.push_back(up);
	}

	vector<UpdateRequest>::iterator it;
	for (it = initial.begin(); it < initial.end(); it++) {
		UpdateRequest req = *it;
    SNode data = req.n.getData();
    int v;
    while (req.w < (v = data.get_dist())) {
      if (data.get_dist() == v ) {
      	data.set_dist(req.w);
//        req.n.map(body, req, ctx, MethodFlag.NONE); //TODO:
        break;
      }
    }
	}

}
