#ifndef DOMAIN_SUPPORT_H
#define DOMAIN_SUPPORT_H
/**
 * Code ported from on below link. Modified under Galois.
 *
 * https://github.com/qcri/Arabesque
 *
 * Copyright (c) 2015 Qatar Computing Research Institute
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

#include "pangolin/gtypes.h"

class DomainSupport {
public:
	DomainSupport() {
		num_domains = 0;
		enough_support = false;
	}
	DomainSupport(unsigned n) {
		num_domains = n;
		enough_support = false;
		domains_reached_support.resize(n);
		std::fill(domains_reached_support.begin(), domains_reached_support.end(), 0);
		domain_sets.resize(n);
	}
	~DomainSupport() {}
	void set_threshold(unsigned minsup) {
		minimum_support = minsup;
		//for (auto i = 0; i < num_domains; i++)
		//	domain_sets[i].get_allocator().allocate(minsup+1);
	}
	void clean() {
		domains_reached_support.clear();
		domain_sets.clear();
	}
	void resize(unsigned n) {
		num_domains = n;
		enough_support = false;
		domains_reached_support.resize(n);
		std::fill(domains_reached_support.begin(), domains_reached_support.end(), 0);
		domain_sets.resize(n);
	}
	bool is_frequent() { return enough_support; }
	void set_frequent() { if (get_support()) enough_support = true; }
	bool has_domain_reached_support(int i) {
		assert(i < num_domains);
		return domains_reached_support[i];
		//return enough_support || domains_reached_support[i];
	}
	void set_domain_frequent(int i) {
		domains_reached_support[i] = 1;
		domain_sets[i].clear();
	}
	void add_vertex(int i, VertexId vid) {
		domain_sets[i].insert(vid);
		if (domain_sets[i].size() >= minimum_support)
			set_domain_frequent(i);
	}
	bool add_vertices(int i, IntSet &vertices) {
		domains_reached_support[i] = 0;
		domain_sets[i].insert(vertices.begin(), vertices.end());
		if (domain_sets[i].size() >= minimum_support) {
			set_domain_frequent(i);
			return true;
		}
		return false;
	}
	// counting the minimal image based support
	inline bool get_support() {
		return std::all_of(domains_reached_support.begin(), domains_reached_support.end(), [](bool v) { return v; });
	}

//private:
	unsigned minimum_support;
	int num_domains;
	bool enough_support;
	BoolVec domains_reached_support;
	IntSets domain_sets;
};

//typedef galois::gstl::Map<InitPattern, DomainSupport> InitMap;
typedef std::map<InitPattern, DomainSupport*> InitMap;
typedef galois::substrate::PerThreadStorage<InitMap> InitMaps;
#endif
