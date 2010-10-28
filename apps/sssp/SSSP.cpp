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
	queue<UpdateRequest> initial;
	for (Graph::neighbor_iterator ii = graph->neighbor_begin(src), ee =
			graph->neighbor_end(src); ii != ee; ++ii) {
		GNode dst = *ii;
		int w = getEdgeData(src, dst);
		UpdateRequest *up = new UpdateRequest(dst, w, w <= delta);
		initial.push(*up);
	}

	while (!initial.empty()) {
		UpdateRequest* req = &initial.front();
		initial.pop();
		SNode data = req->n.getData();
		int v;
		while (req->w < (v = data.get_dist())) {
			if (data.get_dist() == v) {
				data.set_dist(req->w);

				for (Graph::neighbor_iterator ii = graph->neighbor_begin(src), ee =
						graph->neighbor_end(src); ii != ee; ++ii) {
					GNode dst = *ii;
					int d = getEdgeData(req->n, dst);
					int newDist = req->w + d;
					initial.push(*(new UpdateRequest(dst, newDist, d <= delta)));
				}
				break;
			}
		}
		delete req;
	}

}
