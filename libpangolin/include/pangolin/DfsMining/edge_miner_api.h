#pragma once
//#include "gtypes.h"
//#include "embedding_list_dfs.h"
#include "pangolin/domain_support.h"

// Sandslash APIs
template <typename Pattern>
class EdgeMinerAPI {
public:
	EdgeMinerAPI() {}
	~EdgeMinerAPI() {}

	static inline bool toExtend(unsigned, unsigned) {
		return true;
	}

	static inline bool toAdd(BaseEdgeEmbeddingList &, Pattern &, unsigned) {
		//return !is_edge_automorphism(n, emb, pos, src, dst, existed, vertex_set);
		//return (support(emb_list, pattern) >= threshold && is_min(pattern));
		//return (is_frequent(emb_list, pattern) && is_min(pattern));
		return true;
	}

	static inline unsigned getPattern(unsigned, unsigned, VertexId, const BaseEdgeEmbedding &, unsigned) {
		return 0;
	}

	static inline void reduction(UlongAccu &acc) { acc += 1; }

protected:
	static inline bool is_frequent(BaseEdgeEmbeddingList &emb_list, Pattern &pattern, unsigned threshold) {
		if (emb_list.size() < threshold) return false;
		DomainSupport ds(pattern.size()+1);
		ds.set_threshold(threshold);
		unsigned emb_id = 0;
		for (auto cur = emb_list.begin(); cur != emb_list.end(); ++cur) {
			BaseEdgeEmbedding* emb_ptr = &(*cur);
			size_t index = pattern.size() - 1;
			while (emb_ptr != NULL) {
				auto from = pattern[index].from;
				auto to = pattern[index].to;
				auto src = emb_ptr->edge->src;
				auto dst = emb_ptr->edge->dst;
				if (!ds.has_domain_reached_support(to) && to > from) ds.add_vertex(to, dst); //forward edge
				if (!ds.has_domain_reached_support(from) && !emb_ptr->prev) ds.add_vertex(from, src); // last element
				//if (to > from) ds.add_vertex(to, dst); //forward edge
				//if (!emb_ptr->prev) ds.add_vertex(from, src); // last element
				emb_ptr = emb_ptr->prev;
				index--;
			}
			emb_id ++;
			if (emb_id >= threshold) ds.set_frequent();
			if (ds.is_frequent()) return true;
		}
		return false;
	}

	static bool is_canonical(Pattern &pattern) {
		if (pattern.size() == 1) return true;
		CGraph graph_is_min; // canonical graph
		pattern.toGraph(graph_is_min);
		Pattern dfscode_is_min;
		EmbeddingLists2D emb_lists;
		for (size_t vid = 0; vid < graph_is_min.size(); ++ vid) {
			auto vlabel = graph_is_min[vid].label;
			for (auto e = graph_is_min[vid].edges.begin(); e != graph_is_min[vid].edges.end(); ++ e) {
				auto ulabel = graph_is_min[e->dst].label;
				if (vlabel <= ulabel)
					emb_lists[vlabel][ulabel].push(2, &(*e), 0);
			}
		}
		auto fromlabel = emb_lists.begin();
		auto tolabel = fromlabel->second.begin();
		dfscode_is_min.push(0, 1, fromlabel->first, 0, tolabel->first);
		return subgraph_is_min(pattern, dfscode_is_min, graph_is_min, tolabel->second);
	}

	static bool subgraph_is_min(Pattern &orig_pattern, Pattern &pattern, CGraph &cgraph, BaseEdgeEmbeddingList &emb_list) {
		const RMPath& rmpath = pattern.buildRMPath();
		auto minlabel        = pattern[0].fromlabel;
		auto maxtoc          = pattern[rmpath[0]].to;
		// backward
		bool found = false;
		VeridT newto = 0;
		BaseEdgeEmbeddingList emb_list_bck;
		for(size_t i = rmpath.size()-1; i >= 1; -- i) {
			for(size_t j = 0; j < emb_list.size(); ++ j) {
				BaseEdgeEmbedding *cur = &emb_list[j];
				History history(cur);
				auto e1 = history[rmpath[i]];
				auto e2 = history[rmpath[0]];
				if (e1 == e2) continue;
				for (auto e = cgraph[e2->dst].edges.begin(); e != cgraph[e2->dst].edges.end(); ++ e) {
					if (history.hasEdge(e->src, e->dst)) continue;
					if ((e->dst == e1->src) && (cgraph[e1->dst].label <= cgraph[e2->dst].label)) {
						emb_list_bck.push(2, &(*e), cur);
						newto = pattern[rmpath[i]].from;
						found = true;
						break;
					}
				}
			}
		}
		if(found) {
			pattern.push(maxtoc, newto, (LabelT)-1, 0, (LabelT)-1);
			auto size = pattern.size() - 1;
			if (orig_pattern[size] != pattern[size]) return false;
			return subgraph_is_min(orig_pattern, pattern, cgraph, emb_list_bck);
		}

		// forward
		bool flg = false;
		VeridT newfrom = 0;
		EmbeddingLists1D emb_lists_fwd;
		for (size_t n = 0; n < emb_list.size(); ++n) {
			BaseEdgeEmbedding *cur = &emb_list[n];
			History history(cur);
			auto e2 = history[rmpath[0]];
			for (auto e = cgraph[e2->dst].edges.begin(); e != cgraph[e2->dst].edges.end(); ++ e) {
				if (minlabel > cgraph[e->dst].label || history.hasVertex(e->dst)) continue;
				if (flg == false) {
					flg = true;
					newfrom = maxtoc;
				}
				emb_lists_fwd[cgraph[e->dst].label].push(2, &(*e), cur);
			}
		}
		for (size_t i = 0; !flg && i < rmpath.size(); ++i) {
			for (size_t n = 0; n < emb_list.size(); ++n) {
				BaseEdgeEmbedding *cur = &emb_list[n];
				History history(cur);
				auto e1 = history[rmpath[i]];
				for (auto e = cgraph[e1->src].edges.begin(); e != cgraph[e1->src].edges.end(); ++ e) {
					auto dst = e->dst;
					auto &v = cgraph[dst];
					if (e1->dst == dst || minlabel > v.label || history.hasVertex(dst)) continue;
					if (cgraph[e1->dst].label <= v.label) {
						if (flg == false) {
							flg = true;
							newfrom = pattern[rmpath[i]].from;
						}
						emb_lists_fwd[v.label].push(2, &(*e), cur);
					}
				}
			}
		}
		if (flg) {
			auto tolabel = emb_lists_fwd.begin();
			pattern.push(newfrom, maxtoc + 1, (LabelT)-1, 0, tolabel->first);
			auto size = pattern.size() - 1;
			if (orig_pattern[size] != pattern[size]) return false;
			return subgraph_is_min(orig_pattern, pattern, cgraph, tolabel->second);
		} 
		return true;
	} 
};

