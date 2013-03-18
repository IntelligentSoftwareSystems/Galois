#ifndef _UTIL_H_
#define _UTIL_H_

#include <map>
#include <set>
#include "boost/tuple/tuple.hpp"

extern boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readGraph(const char *filename); 

boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readSnapDirectedGraph(const char *filename);

boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readRandomGraph(const char *filename);

extern void freeData(std::map<int, std::set<int>*>* outnbrs, std::map<int, std::set<int>*>* innbrs);

#endif
