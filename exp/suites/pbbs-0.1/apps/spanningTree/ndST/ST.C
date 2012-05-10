#include <iostream>
#include <limits.h>
#include "sequence.h"
#include "gettime.h"
#include "graph.h"
//#include "cilk.h"
#include "unionFind.h"
using namespace std;

struct notMax { bool operator() (int i) {return i < INT_MAX;}};

// Assumes root is negative
inline vindex find(vindex i, vindex* parent) {
  if (parent[i] < 0) return i;
  vindex j = parent[i];     
  if (parent[j] < 0) return j;
  do j = parent[j]; 
  while (parent[j] >= 0);
  vindex tmp = parent[i];
  while((tmp=parent[i])!=j){ parent[i]=j; i=tmp; }
  return j;
}

pair<int*, int> st(edgeArray EA){
  edge* E = EA.E;
  int m = EA.nonZeros;
  int n = EA.numRows;
  vindex *parents = newA(vindex,n);
//  parallel_for (int i=0; i < n; i++) parents[i] = -1;
  parallel_doall(int, i, 0, n) { parents[i] = -1; } parallel_doall_end
  vindex *hooks = newA(vindex,n);
//  parallel_for (int i=0; i < n; i++) hooks[i] = INT_MAX;
  parallel_doall(int, i, 0, n) { hooks[i] = INT_MAX; } parallel_doall_end
  edge* st = newA(edge,m);

  timespec req;
  req.tv_sec = 0;
  req.tv_nsec = 1;  
  
  //edge joins only if acquired lock on both endpoints
//  parallel_for (int i = 0; i < m; i++) {
  parallel_doall(int, i, 0, m)  {
    uint j = 0;
    while(1){
      //if(j++ > 1000) abort();
      vindex u = find(E[i].u,parents);
      vindex v = find(E[i].v,parents);
      if(u == v) break;
      else {
	if(utils::CAS(&hooks[v],INT_MAX,i)){
	  if(utils::CAS(&hooks[u],INT_MAX,i)){
	    if(u == find(u,parents)){ //check that roots didn't change
	      if(v == find(v,parents)){
		//union by rank
		if(parents[u] < parents[v]) swap(u,v);
		parents[v] += parents[u];
		parents[u]=v;
		hooks[v]=INT_MAX;
		break;
	      } else hooks[u] = INT_MAX;
	    } else hooks[v] = INT_MAX;
	  } else hooks[v] = INT_MAX;
	} 

      }
      //sleep if failed too many times
      if(j++ > 25){
	j = 0;
	nanosleep(&req, (struct timespec *) NULL); 
      }      
    }
  } parallel_doall_end
  _seq<int> stIdx = sequence::filter((int*) hooks, n, notMax());
  
  free(parents); free(hooks); free(st); 
  cout<<"nInSt = "<<stIdx.n<<endl;
  return pair<int*,int>(stIdx.A, stIdx.n);
}
