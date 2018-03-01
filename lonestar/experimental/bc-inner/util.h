#ifndef _UTIL_H_
#define _UTIL_H_

#include <map>
#include <set>
#include <limits>
#include "boost/tuple/tuple.hpp"

constexpr unsigned infinity = std::numeric_limits<unsigned>::max() / 2;

extern boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readGraph(const char *filename); 

boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readSnapDirectedGraph(const char *filename);

boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readRandomGraph(const char *filename);

extern void freeData(std::map<int, std::set<int>*>* outnbrs, std::map<int, std::set<int>*>* innbrs);

#endif
