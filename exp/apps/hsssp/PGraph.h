/*
 * PGraph.h
 *
 *  Created on: Aug 10, 2015
 *      Author: rashid
 */
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/LC_CSR_Graph.h"
#include "Galois/Graph/Util.h"
#include "Lonestar/BoilerPlate.h"

#ifndef GDIST_EXP_APPS_HPR_PGRAPH_H_
#define GDIST_EXP_APPS_HPR_PGRAPH_H_



/*********************************************************************************
 * Partitioned graph structure.
 **********************************************************************************/
template<typename GraphTy>
struct pGraph {

   typedef typename GraphTy::edge_data_type EdgeDataType;
   typedef typename GraphTy::node_data_type NodeDataType;
   typedef typename GraphTy::iterator iterator;
   typedef typename GraphTy::edge_iterator edge_iterator;
   typedef unsigned GlobalIDType;
   typedef iterator LocalIDType;

   GraphTy g;
   unsigned g_offset; // LID + g_offset = GID
   unsigned numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes) are replicas
   unsigned numNodes; // number of nodes (may differ from g.size() to simplify loading)
   unsigned numEdges;

   // [numNodes, g.size()) should be ignored
   std::vector<unsigned> m_L2G; // GID = m_L2G[LID - numOwned]
   unsigned id; // my hostid
   std::vector<unsigned> lastNodes; //[ [i - 1], [i]) -> Node owned by host i
   unsigned getHost(GlobalIDType node) { // node is GID
      return std::distance(lastNodes.begin(), std::upper_bound(lastNodes.begin(), lastNodes.end(), node));
   }
   unsigned G2L(GlobalIDType GID) {
      //Locally owned nodes
      if(GID >= g_offset && GID < g_offset+numOwned) return GID-g_offset;
      //Ghost cells.
      auto ii = std::find(m_L2G.begin(), m_L2G.end(), GID);
      assert(ii != m_L2G.end());
      return std::distance(m_L2G.begin(), ii) + numOwned;
   }
   unsigned L2G(iterator LID){
         auto tx_lid = std::distance(g.begin(), LID);
         return  tx_lid < numOwned? tx_lid+g_offset: m_L2G[tx_lid-numOwned];
      }
   pGraph() : g_offset(0), numOwned(0), numNodes(0), id(0), numEdges(0) {
   }
   /********************************************************************
    * Make code more readable by using the following wrappers when iterating
    * over either owned nodes or ghost nodes.
    * ******************************************************************/
   iterator begin() {
      return g.begin();
   }
   iterator end() {
      return g.begin() + numOwned;
   }
   iterator ghost_begin() {
      return g.begin() + numOwned;
   }
   iterator ghost_end() {
      return g.begin() + numNodes;
   }

//   unsigned locam_L2Global(unsigned tx_lid){
//      return  tx_lid < numOwned? tx_lid+g_offset: m_L2G[tx_lid-numOwned];
//   }
   /*
    * Debugging routine. Does not print the m_L2G.
    * */
   void print(){
      auto id= Galois::Runtime::NetworkInterface::ID;
      fprintf(stderr, "H-%d::[g_offset=%d, numOwned=%d, numNodes=%d, numEdges=%d\n", id, g_offset, numOwned, numNodes, numEdges);
      for(auto n = g.begin(); n!=g.begin()+numOwned; ++n){
         for(auto e = g.edge_begin(*n); e!=g.edge_end(*n); ++e){
            auto dst = g.getEdgeDst(e);
            auto wt = g.getEdgeData(e);
            fprintf(stderr, "H-%d::[src:%d, dst:%d, wt:%d]\n", id, *n, dst, wt );
         }
      }
      for (auto n =0; n<lastNodes.size(); ++n){
         fprintf(stderr,"H-%d::[n=%d,lastNodes[n]=%d] ", id, n, lastNodes[n] );
      }
      fprintf(stderr, "m_L2G[");
      for(auto i : m_L2G)
         fprintf(stderr,"%d, ", i );
      fprintf(stderr, "]\n");
   }
   /*********************************************************************************
    * Given a partitioned graph  .
    * lastNodes maintains indices of nodes for each co-host. This is computed by
    * determining the number of nodes for each partition in 'pernum', and going over
    * all the nodes assigning the next 'pernum' nodes to the ith partition.
    * The lastNodes is used to find the host by a binary search.
    **********************************************************************************/

   void loadLastNodes(size_t size, unsigned numHosts) {
      if (numHosts == 1)
         return;

      auto p = Galois::block_range(0UL, size, 0, numHosts);
      unsigned pernum = p.second - p.first;
      unsigned pos = pernum;

      while (pos < size) {
         this->lastNodes.push_back(pos);
         pos += pernum;
      }
   #if _HETERO_DEBUG_
      for (int i = 0; size < 10 && i < size; i++) {
         printf("node %d owned by %d\n", i, this->getHost(i));
      }
   #endif
   }
   /*********************************************************************************
    * Load a partitioned graph from a file.
    * @param file the graph filename to be loaded.
    * @param hostID the identifier of the current host.
    * @param numHosts the total number of hosts.
    * @param out A graph instance that stores the actual graph.
    * @return a partitioned graph backed by the #out instance.
    **********************************************************************************/
   void loadGraph(std::string file) {
      Galois::Graph::FileGraph fg;
      fg.fromFile(file);
      unsigned hostID = Galois::Runtime::NetworkInterface::ID;
      unsigned numHosts = Galois::Runtime::NetworkInterface::Num;
      auto p = Galois::block_range(0UL, fg.size(), hostID, numHosts);
      this->g_offset = p.first;
      this->numOwned = p.second - p.first;
      this->id = hostID;
      std::vector<unsigned> perm(fg.size(), ~0); //[i (orig)] -> j (final)
      unsigned nextSlot = 0;
   //   std::cout << fg.size() << " " << p.first << " " << p.second << "\n";
      //Fill our partition
      for (unsigned i = p.first; i < p.second; ++i) {
         //printf("%d: owned: %d local: %d\n", hostID, i, nextSlot);
         perm[i] = nextSlot++;
      }
      //find ghost cells
      for (auto ii = fg.begin() + p.first; ii != fg.begin() + p.second; ++ii) {
         for (auto jj = fg.edge_begin(*ii); jj != fg.edge_end(*ii); ++jj) {
            //std::cout << *ii << " " << *jj << " " << nextSlot << " " << perm.size() << "\n";
            //      assert(*jj < perm.size());
            auto dst = fg.getEdgeDst(jj);
            int edata  = fg.getEdgeData<int>(jj);
//            std::cout<<" Edge :: "<< *ii <<", "<<dst << ", "<<edata << "\n";
            if (perm.at(dst) == ~0) {
               //printf("%d: ghost: %d local: %d\n", hostID, dst, nextSlot);
               perm[dst] = nextSlot++;
               this->m_L2G.push_back(dst);
            }
         }
      }
      this->numNodes = nextSlot;

      //Fill remainder of graph since permute doesn't support truncating
      for (auto ii = fg.begin(); ii != fg.end(); ++ii)
         if (perm[*ii] == ~0)
            perm[*ii] = nextSlot++;
   //   std::cout << nextSlot << " " << fg.size() << "\n";
      assert(nextSlot == fg.size());
      //permute graph
      Galois::Graph::FileGraph fg2;
      Galois::Graph::permute<EdgeDataType>(fg, perm, fg2);
      Galois::Graph::readGraph(this->g, fg2);

      ///RK: Print graph to debug.
  /*    for(auto n = this->g.begin(); n!=this->g.end(); ++n){
         for(auto e = this->g.edge_begin(*n); e!=this->g.edge_end(*n); ++e){
            std::cout<<"Graph-Edge "<<*n << " , "<<this->g.getEdgeDst(e) << ", "<<this->g.getEdgeData(e) <<"\n";
         }
      }*/

      loadLastNodes(fg.size(), numHosts);

      /* TODO: This still counts edges from ghosts to remote nodes,
       ideally we only want edges from ghosts to local nodes.

       See pGraphToMarshalGraph for one implementation.
       */
      this->numEdges = std::distance(this->g.edge_begin(*this->g.begin()), this->g.edge_end(*(this->g.begin() + this->numNodes - 1)));
//      print();
      return;
   }
};
#endif /* GDIST_EXP_APPS_HPR_PGRAPH_H_ */
