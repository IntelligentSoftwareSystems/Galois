#include "Galois/Endian.h"
#include <boost/tuple/tuple.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#ifdef __linux__
#include <linux/mman.h>
#endif

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <list>

#include "control.h"
#include "ND.h"
#include "ED.h"
#include "BCGraph.h"
#include "util.h"

using namespace std;

#define MAGICNUM 2
#define PRINT_INOUT 0

//File 1 format V1:
//version (1) {uint64_t LE}
//numNodes {uint64_t LE}
//numEdges {uint64_t LE}
//inindexs[numNodes+1] {uint32_t LE} 
//potential padding (32bit max) to Re-Align to 64bits
//inedges[numEdges] {uint32_t LE}
//potential padding (32bit max) to Re-Align to 64bits
//outindexs[numNodes+1] {uint32_t LE} 
//potential padding (32bit max) to Re-Align to 64bits
//
//File 2 format V1:
//version (1) {uint64_t LE}
//outedges[numEdges] {uint32_t LE}
//potential padding (32bit max) to Re-Align to 64bits

int DEF_DISTANCE;
//static const char* help = "(rmat|snap) <input file> <output file name scheme>";

string fname1;
string fname2;
string fname1_BE;
string fname2_BE;

boost::tuple<void*, int, int> 
mapFile(const char *filename) {
  
  int fileFD = open(filename, O_RDONLY);
  struct stat buf;
  fstat(fileFD, &buf);
  size_t fileLength = buf.st_size;

  int _MAP_BASE = MAP_PRIVATE;
#ifdef MAP_POPULATE
  _MAP_BASE  |= MAP_POPULATE;
#endif

  void* m = mmap(0, fileLength, PROT_READ,_MAP_BASE, fileFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    cerr << "Problem with mmap " << filename << endl;
  }
  cerr << "mmaped file " << filename << " of size " << fileLength << endl;
  return boost::make_tuple(m, fileLength, fileFD);

#if defined (__SVR4) && defined (__sun)

#endif

}

void readFromFiles() {

  boost::tuple<void*,int,int> ft1 = mapFile(fname1.c_str());
  void *m = ft1.get<0>();
  uint64_t* fptr = (uint64_t*)m;
  uint64_t version = *fptr++;
  assert(version == 2);
  uint64_t numNodes = *fptr++;
  uint64_t numEdges = *fptr++;
  cerr << "Read numNodes " << numNodes << " numEdges: " << numEdges << endl; 
  int idxArraysLen = numNodes+1;

  uint32_t *fptr32 = (uint32_t *)fptr;
  //int * inIdx = (int *)fptr32;
  fptr32 += idxArraysLen;
  if (idxArraysLen % 2)
    fptr32++;
  //int * ins = (int *)fptr32; 
  fptr32 += numEdges;
  if (numEdges % 2)
    fptr32 += 1;
  //int * outIdx = (int *)fptr32;
  fptr32 += idxArraysLen;
  if (idxArraysLen % 2)
    fptr32++;

  boost::tuple<void*,int,int> ft2 = mapFile(fname2.c_str());
  void *m2 = ft2.get<0>();
  fptr = (uint64_t*)m2;
  version = *fptr++;
  assert(version == 2);
  fptr32 = (uint32_t *)fptr;
  //int * outs = (int *)fptr32; 
  fptr32 += numEdges;
  if (numEdges % 2)
    fptr32 += 1;

#if PRINT_INOUT 
  cerr << outIdx << endl;
  cerr << "outIdx: ";
  for (int i=0; i<numNodes+1; ++i) {
    cerr << outIdx[i] << " ";
  }
  cerr << endl;
  cerr << "outs: ";
  for (int i=0; i<numEdges; ++i) {
    cerr << outs[i] << " ";
  }
  cerr << endl;

  cerr << "inIdx: ";
  for (int i=0; i<numNodes+1; ++i) {
    cerr << inIdx[i] << " ";
  }
  cerr << endl;
  cerr << "ins: ";
  for (int i=0; i<numEdges; ++i) {
    cerr << ins[i] << " ";
  }
  cerr << endl;
#endif

  munmap(m2, ft2.get<1>());
  close(ft2.get<2>());
  munmap(m, ft1.get<1>());
  close(ft1.get<2>());
}

void readFromFilesChangeEndianessAndWrite() {

	ofstream fout(fname1_BE.c_str(), ios::binary);

  boost::tuple<void*,int,int> ft1 = mapFile(fname1.c_str());
  void *m = ft1.get<0>();
  uint64_t* fptr = (uint64_t*)m;
  uint64_t version = *fptr++;
  uint64_t sw_version = galois::bswap64(version);
  fout.write((char*)(&sw_version), sizeof(sw_version));
  assert(version == 2);
  uint64_t numNodes = *fptr++;
  uint64_t numEdges = *fptr++;
  uint64_t sw_numNodes = galois::bswap64(numNodes);
  fout.write((char*)(&sw_numNodes), sizeof(sw_numNodes));
  uint64_t sw_numEdges = galois::bswap64(numEdges);
  fout.write((char*)(&sw_numEdges), sizeof(sw_numEdges));

  cerr << "Read numNodes " << numNodes << " numEdges: " << numEdges << endl; 
  int idxArraysLen = numNodes+1;

  uint32_t *fptr32 = (uint32_t *)fptr;
  int * inIdx = (int *)fptr32;
  fptr32 += idxArraysLen;

	for (int i=0; i<idxArraysLen; ++i) {
	  uint32_t sw_inIdx = galois::bswap32(inIdx[i]);
  	fout.write((char*)(&sw_inIdx), sizeof(sw_inIdx));
	}

  if (idxArraysLen % 2) {
    uint32_t t = galois::bswap32(1); //dummy
    fout.write((char*)(&t), sizeof(t));
		fptr32++;
	}

  int * ins = (int *)fptr32; 
  fptr32 += numEdges;
	for (unsigned int i=0; i<numEdges; ++i) {
          uint32_t sw_ins = galois::bswap32(ins[i]);
		fout.write((char*)(&sw_ins), sizeof(sw_ins));
	}
  if (numEdges % 2) {
    uint32_t t = galois::bswap32(1); //dummy
    fout.write((char*)(&t), sizeof(t));
		fptr32 += 1;
	}

  int * outIdx = (int *)fptr32;
  fptr32 += idxArraysLen;
	for (int i=0; i<idxArraysLen; ++i) {
		uint32_t sw_outIdx = galois::bswap32(outIdx[i]);
		fout.write((char*)(&sw_outIdx), sizeof(sw_outIdx));
	}
  if (idxArraysLen % 2) {
    uint32_t t = galois::bswap32(1); //dummy
    fout.write((char*)(&t), sizeof(t));
		fptr32++;
	}
	fout.close();

	ofstream fout2(fname2_BE.c_str(), ios::binary);
  boost::tuple<void*,int,int> ft2 = mapFile(fname2.c_str());
  void *m2 = ft2.get<0>();
  fptr = (uint64_t*)m2;
  version = *fptr++;
  sw_version = galois::bswap64(version);
  fout2.write((char*)(&sw_version), sizeof(sw_version));
  assert(version == 2);
  fptr32 = (uint32_t *)fptr;
  int * outs = (int *)fptr32;
  fptr32 += numEdges;
	for (unsigned int i=0; i<numEdges; ++i) {
		uint32_t sw_outs = galois::bswap32(outs[i]);
		fout2.write((char*)(&sw_outs), sizeof(sw_outs));
	}
  if (numEdges % 2) {
    uint32_t t = galois::bswap32(1); //dummy
    fout2.write((char*)(&t), sizeof(t));
		fptr32 += 1;
	}
	fout2.close();
#if PRINT_INOUT 
  cerr << outIdx << endl;
  cerr << "outIdx: ";
  for (int i=0; i<numNodes+1; ++i) {
    cerr << outIdx[i] << " ";
  }
  cerr << endl;
  cerr << "outs: ";
  for (int i=0; i<numEdges; ++i) {
    cerr << outs[i] << " ";
  }
  cerr << endl;

  cerr << "inIdx: ";
  for (int i=0; i<numNodes+1; ++i) {
    cerr << inIdx[i] << " ";
  }
  cerr << endl;
  cerr << "ins: ";
  for (int i=0; i<numEdges; ++i) {
    cerr << ins[i] << " ";
  }
  cerr << endl;
#endif

  munmap(m2, ft2.get<1>());
  close(ft2.get<2>());
  munmap(m, ft1.get<1>());
  close(ft1.get<2>());
}



void writeToFile(int nnodes, int nedges, int nInIdx, int *inIdx, int nins, 
    int * ins, int nOutIdx, int *outIdx, vector<int> *outs) {
  uint64_t magic = MAGICNUM;
  ofstream fout(fname1.c_str(), ios::binary);
  fout.write((char*)(&magic), sizeof(magic));
  uint64_t numNodes = (uint64_t)nnodes;
  uint64_t numEdges = (uint64_t)nedges;
  fout.write((char*)(&numNodes), sizeof(numNodes));
  fout.write((char*)(&numEdges), sizeof(numEdges));
  
  for (int i=0; i<nInIdx; ++i) {
    uint32_t t = (uint32_t)inIdx[i];
    fout.write((char*)(&t), sizeof(t));
  }
  if (nInIdx % 2) {
    uint32_t t = 1; //dummy
    fout.write((char*)(&t), sizeof(t));
  } 
  for (int i=0; i<nins; ++i) {
    uint32_t t = (uint32_t)ins[i];
    fout.write((char*)(&t), sizeof(t));
  }
  if (nins % 2) {
    uint32_t t = 1; //dummy
    fout.write((char*)(&t), sizeof(t));
  } 
  for (int i=0; i<nOutIdx; ++i) {
    uint32_t t = (uint32_t)outIdx[i];
    fout.write((char*)(&t), sizeof(t));
  }
  if (nOutIdx % 2) {
    uint32_t t = 1; //dummy
    fout.write((char*)(&t), sizeof(t));
  }
  fout.close();

  ofstream fout1(fname2.c_str(), ios::binary);
  fout1.write((char*)(&magic), sizeof(magic));
  for (vector<int>::const_iterator it = outs->begin(); it != outs->end(); ++it) {
    uint32_t t = (uint32_t)(*it);
    fout1.write((char*)(&t), sizeof(t));
  }
  if (outs->size() % 2) {
    uint32_t t = 1; //dummy
    fout1.write((char*)(&t), sizeof(t));
  } 
  fout1.close();

}

void construct(int nnodes, int nedges, const std::map<int, std::set<int>*> & _succs, 
    const std::map<int, std::set<int>*> & _preds) {

  ND * nodes = new ND[nnodes];
  ED * edgeData = new ED[nedges];
  int * inIdx = new int[nnodes+1];

  int * ins = new int[nedges];
  int * outIdx = new int[nnodes+1];
  int ninIdx = nnodes+1;
  int nins = nedges; 
  int noutIdx = nnodes+1;
  vector<int> * outs = new vector<int>();

  for (int i =0; i<nnodes; ++i) {
    nodes[i].id = i;
  }

  int outIdxCnt = 0;
  outIdx[outIdxCnt++] = 0;
  int edCnt = 0;
  for (int nc=0; nc<nnodes; ++nc) {
    std::map<int, std::set<int>*>::const_iterator it1 = _succs.find(nc);
    if (it1 != _succs.end()) {
      int n = it1->first;
      std::set<int> *succsOfN = it1->second;
      for (std::set<int>::const_iterator it2 = succsOfN->begin(); it2 != succsOfN->end(); ++it2) {
        outs->push_back(*it2);
        edgeData[edCnt].src = &(nodes[n]);
        edgeData[edCnt].dst = &(nodes[*it2]);
        edCnt++;
      }
    }
    outIdx[outIdxCnt++] = edCnt;
  }
  int insCnt = 0;
  int inIdxCnt = 0;
  inIdx[inIdxCnt++] = 0;
  for (int nc=0; nc<nnodes; ++nc) {
    std::map<int, std::set<int>*>::const_iterator it1 = _preds.find(nc);
    if (it1 != _preds.end()) {
      int nid = it1->first;
      std::set<int> *predsOfN = it1->second;
      for (std::set<int>::const_iterator it2 = predsOfN->begin(); it2 != predsOfN->end(); ++it2) {
        int predId = *it2;
        int start = outIdx[predId];
        int end = outIdx[predId + 1];
        //std::cerr << " Scanning edges from " << start << " to " << end << " for pred " << predId << std::endl;
        bool found = false;
        for (int i = start; i < end; i++) {
          const ED & e = edgeData[i];

          assert (e.src->id == predId);
          if (e.dst->id == nid) {
            ins[insCnt++] = i;
            found = true;
            break;
          }
        }
        /*if (!found) {
          std::cerr << "Problem trying to find predecessor edge bw " << predId << " and " << nid << std::endl;
          for (std::set<ND*>::const_iterator it3 = predsOfN->begin(); it3 != predsOfN->end(); ++it3) {
          int predId = (*it3)->id;
          std::cerr << predId << std::endl;
          }
          }*/
        assert (found);
      }
    }
    inIdx[inIdxCnt++] = insCnt;
  }
  std::cerr << "_outIdx " << outIdxCnt << " nnodes " << nnodes << " _inIdx " 
    << inIdxCnt << " _ins " << insCnt << " outs " << outs->size() << std::endl;

  writeToFile(nnodes, nedges, ninIdx, inIdx, nins, ins, noutIdx, outIdx, outs);
#if PRINT_INOUT
  cerr << outIdx << endl;
  cerr << "outIdx: ";
  for (int i=0; i<noutIdx; ++i) {
    cerr << outIdx[i] << " ";
  }
  cerr << endl;
  cerr << "outs: ";
  for (vector<int>::const_iterator it = outs->begin(); it != outs->end(); ++it) 
    cerr << *it << " ";
  
  cerr << endl;

  cerr << "inIdx: ";
  for (int i=0; i<ninIdx; ++i) {
    cerr << inIdx[i] << " ";
  }
  cerr << endl;
  cerr << "ins: ";
  for (int i=0; i<nins; ++i) {
    cerr << ins[i] << " ";
  }
  cerr << endl;
#endif
  delete [] inIdx;
  delete [] ins;
  delete [] outIdx;
  delete [] edgeData;
  delete [] nodes;
  delete outs;
}

int main(int argc, const char** argv) {
  
#if CONCURRENT
  cerr << "Running in concurrent mode" << endl;
#else
  cerr << "Running in serial mode" << endl;
#endif
//	vector<const char*> args = parse_command_line(argc, argv, help);
	
	if (std::string("gen").compare(argv[1]) == 0) {
		boost::tuple<int, int, map<int, set<int>*>*, map<int, set<int>*>*> p; 
		if (std::string("rmat").compare(argv[2]) == 0) {
			p = readGraph(argv[3]);
		} else if (std::string("snap").compare(argv[2]) == 0) {
			p = readSnapDirectedGraph(argv[3]);
		} else if (std::string("rand").compare(argv[2]) == 0) {
			p = readRandomGraph(argv[3]);  
		}

		map<int, set<int>*>* outnbrs = p.get<2>();
		map<int, set<int>*>* innbrs = p.get<3>();
		uint64_t nnodes = p.get<0>();
		uint64_t nedges = p.get<1>();

		string tmp(argv[4]);
		fname1 = tmp + "1.gr";
		fname2 = tmp + "2.gr";	

		cerr << "Writing to files " << fname1 << " " << fname2 << endl;
		construct(nnodes, nedges, *outnbrs, *innbrs);
		freeData(outnbrs, innbrs);
		readFromFiles();
	} else if (std::string("conv").compare(argv[1]) == 0) {
		string tmp(argv[2]);
		fname1 = tmp + "1.gr";
		fname2 = tmp + "2.gr";	
	  fname1_BE = tmp + "BE_1.gr";
  	fname2_BE = tmp + "BE_2.gr";
		cerr << "Writing to files " << fname1_BE << " " << fname2_BE << endl;
		readFromFilesChangeEndianessAndWrite();
	} else {
		std::cerr << " Specify action (\"gen\"|\"conv\")" << std::endl;
	}
  return 0;
}


