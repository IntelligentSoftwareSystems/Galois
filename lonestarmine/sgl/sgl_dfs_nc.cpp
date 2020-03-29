#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

const char* name = "Sgl";
const char* desc = "Listing vertex-induced subgraphs of a single arbitrary pattern in a graph using DFS traversal";
const char* url  = 0;

//#define DIAMOND
#define RECTANGLE // 4-cycle
#ifdef RECTANGLE
#define input_ccode 2
#else
#define input_ccode 3
#endif
#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI {
public:
	static inline bool toExtend(unsigned level, unsigned pos, const std::vector<VertexId> *emb) {
		return pos == level; // only extend the last vertex
	}
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId vid, 
		unsigned src_idx, BYTE ccode, const std::vector<VertexId> *emb) { 
		if (vid == (*emb)[0]) return false;
		if (vid == (*emb)[1]) return false;
		if (level == 1) {
#ifdef RECTANGLE
			if (vid < (*emb)[0]) return false;
#else
#ifdef DIAMOND
			if (vid < (*emb)[0] && vid < (*emb)[1]) return false;
#endif
#endif
			if ((input_ccode & 3) == 3) {
				if (ccode == 3) return true;
				else return false;
			} else {
				if (ccode == 2) return true;
				else return false;
			}
		} else if (level == 2) {
			//if (vid == (*emb)[2]) return false;
#ifdef RECTANGLE
			if (vid < (*emb)[0] || vid < (*emb)[1]) return false;
			if (ccode == 5) return true; // 4-cycle
#else
#ifdef DIAMOND
			if (ccode != 5 && ccode != 6) return false;
			if (ccode == 5 && ((*emb)[2] < (*emb)[0] || vid < (*emb)[1])) return false;
			if (ccode == 6 && ((*emb)[2] < (*emb)[1] || vid < (*emb)[0])) return false;
			return true;
#else
			if (ccode == 4) return true; // tailed_tri
			else return false;
#endif // end diamond
#endif // end 4cycle
		} else {
			exit(1);
		}
		return false; 
	}
/*
	static inline unsigned getPattern(unsigned level, unsigned max_level, VertexId src, 
		VertexId dst, BYTE ccode, unsigned pcode, BYTE src_idx, std::vector<VertexId> *emb) { 
		return get_pattern_id(level, dst, ccode, pcode, src_idx);
	}
	bool is_tailedtri(BYTE src_idx, BYTE ccode, unsigned pcode) {
		if (pcode == 0 && ccode < 5 && ccode != 3) return true; 
		if (pcode == 1) {
			if (src_idx == 0) {
				if (ccode == 3 || ccode == 5) return true;
			} else {
				if (ccode == 3 || ccode == 6) return true;
			}
		}
		return false;
	}
	bool is_diamond(BYTE src_idx, BYTE ccode, unsigned pcode) {
		if (pcode == 0) { // extending a triangle
			if (ccode == 3 || ccode == 5 || ccode == 6) {
				return true; // diamond
			}
		} else if (ccode == 7) return true; // diamond
		return false;
	}
	bool is_4cycle(BYTE src_idx, BYTE ccode, unsigned pcode) {
		if (pcode == 1) { // extending a wedge
			if (src_idx == 0) {
				if (ccode == 6) return true; // 4-cycle
			} else {
				if (ccode == 5) return true; // 4-cycle
			}
		}
		return false;
	}
	bool is_3star(BYTE src_idx, BYTE ccode, unsigned pcode) {
		if (pcode == 1) { // extending a wedge
			if (src_idx == 0) {
				if (ccode == 1) return true; // 3-star
			} else {
				if (ccode == 2) return true; // 3-star
			}
		}
		return false;
	}
*/
};

class AppMiner : public VertexMinerDFS<MyAPI, false, true, true, false, true, false, true, false> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<MyAPI, false, true, true, false, true, false, true, false>(ms, nt) {
		assert(k > 2);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"

