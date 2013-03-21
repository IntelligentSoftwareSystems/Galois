#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include "util.h"
#include "control.h"

extern int DEF_DISTANCE;

boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readRandomGraph(const char *filename) {
  
  std::ifstream infile;
  infile.open(filename, std::ifstream::in); // opens the vector file
  if (!infile) { // file couldn't be opened
    std::cerr << "file " << filename << " could not be opened" << std::endl;
    //abort();
  } 
  std::string linebuf;
  std::map<int, std::set<int>*> *outnbrs = new std::map<int, std::set<int>*>();
  std::map<int, std::set<int>*> *innbrs = new std::map<int, std::set<int>*>();
	int nnodes = 0, nedges = 0;
	while (!std::getline(infile, linebuf).eof()) {
		//cerr << linebuf << endl;
		if (linebuf[0] == 'c') {
			continue; // Skip the lines that start with comments
    } else if (linebuf[0] == 'p') { 
      std::stringstream slb(linebuf);
      std::string tmp1, tmp2;
      slb >> tmp1 >> tmp2 >> nnodes >> nedges;
      DEF_DISTANCE = 2*nnodes;
      continue;
    }	else if (linebuf[0] == 'a') {
      std::string tmp;
      int from, to, weight;
      std::stringstream slb(linebuf);
      slb >> tmp >> from >> to >> weight;
      //std::cerr << tmp << " ## " << from << " " << to << " " << weight << std::endl;
      std::set<int> *outNbrlist; 
      std::map<int, std::set<int>*>::iterator it = outnbrs->find(from);
      if (it == outnbrs->end()) {
        outNbrlist = new std::set<int>();	
        outnbrs->insert(std::make_pair(from, outNbrlist));
      } else {
        outNbrlist = it->second;
      }
      assert(to <= nnodes);
      outNbrlist->insert(to);

      std::set<int> *inNbrlist; 
      std::map<int, std::set<int>*>::iterator it1 = innbrs->find(to);
      if (it1 == innbrs->end()) {
        inNbrlist = new std::set<int>();	
        innbrs->insert(std::make_pair(to, inNbrlist));
      } else {
        inNbrlist = it1->second;
      }
      assert(from <= nnodes);
      inNbrlist->insert(from);
    } else {
      std::cerr << "Unknown line: " << linebuf << std::endl;
      assert(false);
    }
  }

  int actualEdgNum = 0;
  for (std::map<int, std::set<int>*>::const_iterator it = outnbrs->begin(); it != outnbrs->end(); ++it) {
    actualEdgNum += it->second->size(); 
  }
  //if (actualEdgNum != nedges) {
  //  std::cerr << "Problem finding right number of edges " << actualEdgNum << " vs. " << nedges << std::endl;
  //  assert(false);
  //}
  std::cerr << "Finished reading graph " << nnodes << " " << actualEdgNum << std::endl;
  infile.close();
#if VERBOSE_GRAPH_READ
  std::cerr << "Read graph is: " << std::endl;
  for (std::map<int, std::set<int>*>::const_iterator it = outnbrs->begin(); it != outnbrs->end(); ++it) {
    std::cerr << "Node " << it->first << std::endl << it->second->size() <<  " Successors: ";
    for (std::set<int>::const_iterator it1 = it->second->begin(); it1 != it->second->end(); ++it1) {
      std::cerr << (*it1) << " ";
    }
    std::cerr << std::endl;
  }
  for (std::map<int, std::set<int>*>::const_iterator it = innbrs->begin(); it != innbrs->end(); ++it) {
    std::cerr << "Node " << it->first << std::endl << it->second->size() << " Predecessors: ";
    for (std::set<int>::const_iterator it1 = it->second->begin(); it1 != it->second->end(); ++it1) {
      std::cerr << (*it1) << " ";
    }
    std::cerr << std::endl;
  }
#endif
	return boost::make_tuple(nnodes, actualEdgNum, outnbrs, innbrs);
}

boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readSnapDirectedGraph(const char *filename) {
  
  std::ifstream infile;
  infile.open(filename, std::ifstream::in); // opens the vector file
  if (!infile) { // file couldn't be opened
    std::cerr << "file " << filename << " could not be opened" << std::endl;
    //abort();
  } 
  std::string linebuf;
  std::map<int, std::set<int>*> *outnbrs = new std::map<int, std::set<int>*>();
  std::map<int, std::set<int>*> *innbrs = new std::map<int, std::set<int>*>();
  int maxNodeId = 0;
	while (!std::getline(infile, linebuf).eof()) {
		//cerr << linebuf << endl;
		if (linebuf[0] == '#')
			continue; // Skip the first lines that start with comments
		
		int from, to;
    std::stringstream slb(linebuf);
		slb >> from >> to;

    if (from > maxNodeId)
      maxNodeId = from;
    if (to > maxNodeId)
      maxNodeId = to;

    std::set<int> *outNbrlist; 
    std::map<int, std::set<int>*>::iterator it = outnbrs->find(from);
    if (it == outnbrs->end()) {
      outNbrlist = new std::set<int>();	
      outnbrs->insert(std::make_pair(from, outNbrlist));
    } else {
      outNbrlist = it->second;
    }
    outNbrlist->insert(to);

    std::set<int> *inNbrlist; 
    std::map<int, std::set<int>*>::iterator it1 = innbrs->find(to);
    if (it1 == innbrs->end()) {
      inNbrlist = new std::set<int>();	
      innbrs->insert(std::make_pair(to, inNbrlist));
    } else {
      inNbrlist = it1->second;
    }
    inNbrlist->insert(from);
	}

  int nnodes = maxNodeId+1;
  DEF_DISTANCE = 2*nnodes;

  int actualEdgNum = 0;
  for (std::map<int, std::set<int>*>::const_iterator it = outnbrs->begin(); it != outnbrs->end(); ++it) {
    actualEdgNum += it->second->size(); 
  }  
  std::cerr << "Finished reading graph " << nnodes << " " << actualEdgNum << std::endl;
  infile.close();
#if VERBOSE_GRAPH_READ
  std::cerr << "Read graph is: " << std::endl;
  for (std::map<int, std::set<int>*>::const_iterator it = outnbrs->begin(); it != outnbrs->end(); ++it) {
    std::cerr << "Node " << it->first << std::endl << it->second->size() <<  " Successors: ";
    for (std::set<int>::const_iterator it1 = it->second->begin(); it1 != it->second->end(); ++it1) {
      std::cerr << (*it1) << " ";
    }
    std::cerr << std::endl;
  }
  for (std::map<int, std::set<int>*>::const_iterator it = innbrs->begin(); it != innbrs->end(); ++it) {
    std::cerr << "Node " << it->first << std::endl << it->second->size() << " Predecessors: ";
    for (std::set<int>::const_iterator it1 = it->second->begin(); it1 != it->second->end(); ++it1) {
      std::cerr << (*it1) << " ";
    }
    std::cerr << std::endl;
  }
#endif
	return boost::make_tuple(nnodes, actualEdgNum, outnbrs, innbrs);
}

boost::tuple<int, int, std::map<int, std::set<int>*> *, std::map<int, std::set<int>*>*>
readGraph(const char *filename) {

	std::ifstream infile;
	infile.open(filename, std::ifstream::in); // opens the vector file
	if (!infile) { // file couldn't be opened
    std::cerr << "file " << filename << " could not be opened" << std::endl;
		//abort();
	}

  std::string linebuf;
	int nnodes = 0, nedges = 0, edgeNum = 0;
	bool firstLineAfterComments = true;
  std::map<int, std::set<int>*> *outnbrs = new std::map<int, std::set<int>*>();
  std::map<int, std::set<int>*> *innbrs = new std::map<int, std::set<int>*>();
	while (!std::getline(infile, linebuf).eof()) {
		//cerr << linebuf << endl;
		if (linebuf[0] == '%')
			continue; // Skip the first lines that start with comments
		if (firstLineAfterComments) {
      std::stringstream(linebuf) >> nnodes >> nedges;
			//cerr << nnodes << " " << nedges << endl;
			DEF_DISTANCE = 2*nnodes;
			for (int i=0; i<nnodes; ++i) {
				outnbrs->insert(std::make_pair(i, new std::set<int>()));
				innbrs->insert(std::make_pair(i, new std::set<int>()));
			}
			firstLineAfterComments = false;
			continue;
		}
		int nid, nedg;
    std::stringstream slb(linebuf);
		slb >> nid >> nedg;
    std::set<int> *outNbrlist = outnbrs->find(nid)->second;
		for (int j=0; j<nedg; ++j) {
      edgeNum++;
			int nbrId, edgW;
			slb >> nbrId >> edgW;
			outNbrlist->insert(nbrId);
			innbrs->find(nbrId)->second->insert(nid);
		}
	}
  int actualEdgNum = 0;
  for (std::map<int, std::set<int>*>::const_iterator it = outnbrs->begin(); it != outnbrs->end(); ++it) {
    actualEdgNum += it->second->size(); 
  }  
  std::cerr << "Finished reading graph " << nnodes << " " << nedges << " "  << edgeNum << " " << actualEdgNum << std::endl;
  infile.close();
#if VERBOSE_GRAPH_READ
  std::cerr << "Read graph is: " << std::endl;
  for (std::map<int, std::set<int>*>::const_iterator it = outnbrs->begin(); it != outnbrs->end(); ++it) {
    std::cerr << "Node " << it->first << std::endl << it->second->size() <<  " Successors: ";
    for (std::set<int>::const_iterator it1 = it->second->begin(); it1 != it->second->end(); ++it1) {
      std::cerr << (*it1) << " ";
    }
    std::cerr << std::endl;
  }
  for (std::map<int, std::set<int>*>::const_iterator it = innbrs->begin(); it != innbrs->end(); ++it) {
    std::cerr << "Node " << it->first << std::endl << it->second->size() << " Predecessors: ";
    for (std::set<int>::const_iterator it1 = it->second->begin(); it1 != it->second->end(); ++it1) {
      std::cerr << (*it1) << " ";
    }
    std::cerr << std::endl;
  }
#endif

	return boost::make_tuple(nnodes, actualEdgNum, outnbrs, innbrs);
}


void freeData(std::map<int, std::set<int>*>* outnbrs, std::map<int, std::set<int>*>* innbrs) {
	for (std::map<int, std::set<int>*>::iterator it = outnbrs->begin(); it != outnbrs->end(); ++it) {
		delete it->second;
	}
  delete outnbrs;
	for (std::map<int, std::set<int>*>::iterator it = innbrs->begin(); it != innbrs->end(); ++it) {
		delete it->second;
	}
  delete innbrs;
}

