#pragma once
#include "pangolin/types.h"

struct SEdge {
	VertexId src;
	VertexId dst;
	SEdge() {}
	SEdge(VertexId _src, VertexId _dst) : src(_src), dst(_dst) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << src << "," << dst << ")";
		return ss.str();
	}
};

