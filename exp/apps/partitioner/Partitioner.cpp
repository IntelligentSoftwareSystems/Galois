/*
 * Partitioner.cpp
 *
 *  Created on: Apr 14, 2016
 *      Author: rashid
 */
/** Partitioner -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Perform vertex-cut partitions on Galois graphs.
 * Includes verification routines.
 *
 * @author Rashid Kaleem (rashid.kaleem@gmail.com)
 */

#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>

#include "Galois/Graphs/FileGraph.h"
#include "Galois/Dist/OfflineGraph.h"

static const char* const name = "Off-line graph partitioner";
static const char* const desc = "A collection of routines to partition graphs off-line.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> numPartitions("num", cll::desc("Number of partitions to be created"), cll::init(2));

typedef OfflineGraph GraphType;
typedef GraphType::edge_iterator EdgeItType;
typedef GraphType::iterator NodeItType;

//using namespace std;
/******************************************************************
 *
 *****************************************************************/
std::string getPartitionFileName(std::string & basename, size_t hostID, size_t num_hosts){
   std::string result = basename;
   result+= ".PART.";
   result+=std::to_string(hostID);
   result+= ".OF.";
   result+=std::to_string(num_hosts);
   return result;
}
std::string getMetaFileName(std::string & basename, size_t hostID, size_t num_hosts){
   std::string result = basename;
   result+= ".META.";
   result+=std::to_string(hostID);
   result+= ".OF.";
   result+=std::to_string(num_hosts);
   return result;
}
/******************************************************************
 *
 *****************************************************************/
struct VertexCutInfo {
   std::vector<size_t> edgeOwners;
   std::vector<size_t> edgesPerHost;
   std::map<NodeItType, std::set<size_t>> vertexOwners;
   std::vector<std::vector<size_t>> verticesPerHost;
   std::vector<std::map<size_t, size_t>> hostGlobalToLocalMapping;

   void init(size_t ne, size_t numHosts) {
      edgeOwners.resize(ne);
      verticesPerHost.resize(numHosts);
      edgesPerHost.resize(numHosts);
      hostGlobalToLocalMapping.resize(numHosts);
   }
   /*
    *
    * */
   void assignEdge(OfflineGraph & g, NodeItType & src, EdgeItType & e, size_t owner) {
      size_t eIdx = std::distance(g.edge_begin(*g.begin()), e);
      auto dst = g.getEdgeDst(e);
      edgeOwners[eIdx] = owner;
      edgesPerHost[owner]++;
      if (vertexOwners[src].insert(owner).second) {
         verticesPerHost[owner].push_back(*src);
         hostGlobalToLocalMapping[owner][*src] = verticesPerHost[owner].size() - 1;
      }
      if (vertexOwners[dst].insert(owner).second) {
         verticesPerHost[owner].push_back(dst);
         hostGlobalToLocalMapping[owner][dst] = verticesPerHost[owner].size() - 1;
      }
   }
   size_t getEdgeOwner(OfflineGraph & g, EdgeItType & e) const {
      size_t eIdx = std::distance(g.edge_begin(*g.begin()), e);
      return edgeOwners[eIdx];
   }
};
/******************************************************************
 *
 *****************************************************************/
struct Partitioner {
   /*
    * Overload this method for different implementations of the partitioning.
    * */
   size_t getEdgeOwner(size_t src, size_t dst, size_t num) {
      return rand() % num;
   }
   /*
    * Partitioning routine.
    * */
   void operator()(std::string & basename, OfflineGraph & g, VertexCutInfo & vcInfo, size_t num_hosts) {
//      const int num_hosts = 4;
      std::cout << "Partitioning for " << num_hosts << " partitions.\n";
      vcInfo.init(g.sizeEdges(), num_hosts);
      for (auto n = g.begin(); n != g.end(); ++n) {
         auto src = *n;
         for (auto nbr = g.edge_begin(*n); nbr != g.edge_end(*n); ++nbr) {
            auto dst = g.getEdgeDst(nbr);
            size_t owner = getEdgeOwner(src, dst, num_hosts);
            vcInfo.assignEdge(g, n, nbr, owner);
         }
      }
      assignVertices(g, vcInfo, num_hosts);
      writePartitions(basename, g, vcInfo, num_hosts);
   }
   /*
    * Edges have been assigned. Now, go over each partition, for any vertex in the partition
    * create a new local-id, and update all the edges to the new local-ids.
    * */
   void assignVertices(OfflineGraph & g, VertexCutInfo & vcInfo, size_t num_hosts) {
      size_t verticesSum = 0;
      if(false){
         for (size_t h = 0; h < num_hosts; ++h) {
            for (size_t v = 0; v < vcInfo.verticesPerHost[h].size(); ++v) {
               auto vertex = vcInfo.verticesPerHost[h][v];
               std::cout << "Host " << h << " Mapped Global:: " << vertex << " to Local:: " << vcInfo.hostGlobalToLocalMapping[h][vertex] << "\n";
            }
         }

      }
      std::vector<size_t> hostVertexCounters(num_hosts);
      for (auto i : vcInfo.vertexOwners) {
         verticesSum += i.second.size();
      }
      for (size_t i = 0; i < num_hosts; ++i) {
         std::cout << "Host :: " << i << " , Vertices:: " << vcInfo.verticesPerHost[i].size() << ", Edges:: " << vcInfo.edgesPerHost[i] << "\n";
      }
      std::cout << "Vertices - Created ::" << verticesSum << " , Actual :: " << g.size() << ", Ratio:: " << verticesSum / (float) (g.size()) << "\n";
   }
   /*
    * Write both the metadata as well as the partition information.
    * */
   void writePartitions(std::string & basename, OfflineGraph & g, VertexCutInfo & vcInfo, size_t num_hosts) {
      //Create graph
      //TODO RK - Fix edgeData
      std::vector<std::vector<std::pair<size_t, size_t>>>newEdges(num_hosts);
      for (auto n = g.begin(); n != g.end(); ++n) {
         auto src = *n;
         for (auto e = g.edge_begin(*n); e != g.edge_end(*n); ++e) {
            auto dst = g.getEdgeDst(e);
            size_t owner = vcInfo.getEdgeOwner(g, e);
            size_t new_src = vcInfo.hostGlobalToLocalMapping[owner][src];
            size_t new_dst = vcInfo.hostGlobalToLocalMapping[owner][dst];
            newEdges[owner].push_back(std::pair<size_t, size_t>(new_src, new_dst));
         }
      }
      for (size_t i = 0; i < num_hosts; ++i) {
         using namespace Galois::Graph;
         FileGraphWriter newGraph;
         newGraph.setNumNodes(vcInfo.hostGlobalToLocalMapping[i].size());
         newGraph.setNumEdges(newEdges[i].size());
         newGraph.phase1();
//         char filename[256];
//         sprintf(filename, "partition_%zu_%zu.dimacs", i, num_hosts);
         std::string meta_file_name = getMetaFileName(basename, i, num_hosts);
//         char meta_file_name[256];
//         sprintf(meta_file_name, "partition_%zu_of_%zu.gr.meta", i, num_hosts);
         std::ofstream meta_file(meta_file_name, std::ofstream::binary);
         auto numEntries = vcInfo.hostGlobalToLocalMapping[i].size();
         meta_file.write(reinterpret_cast<char*>(&(numEntries)), sizeof(numEntries));
         for (auto n : vcInfo.hostGlobalToLocalMapping[i]) {
            auto owner = *vcInfo.vertexOwners[n.second].begin();
            meta_file.write(reinterpret_cast<const char*>(&n.first), sizeof(n.first));
            meta_file.write(reinterpret_cast<const char*>(&n.second), sizeof(n.second));
            meta_file.write(reinterpret_cast<const char*>(&owner), sizeof(owner));
         }
         meta_file.close();

         for (auto e : newEdges[i]) {
            newGraph.incrementDegree(e.first);
         }
         newGraph.phase2();
         for (auto e : newEdges[i]) {
            newGraph.addNeighbor(e.first, e.second);
         }
         newGraph.finish<void>();
//         char gFileName[256];
//         sprintf(gFileName, "partition_%zu_of_%zu.gr", i, num_hosts);
         std::string gFileName = getPartitionFileName(basename, i, num_hosts);
         newGraph.toFile(gFileName);
      }
   }
};

/******************************************************************
 *
 * To verify the partitioning -
 * 1) Read all the graphs individually.
 * 2) Read the partitioning information associated with each sub-graph.
 * 3) Go over each graph and verify that the edges are found in original graph
 *    and sum of edges equals the original graph.
 *****************************************************************/
struct NodeInfo {
   NodeInfo() :
         local_id(0), global_id(0), owner_id(0) {
   }
   NodeInfo(size_t l, size_t g, size_t o) :
         local_id(l), global_id(g), owner_id(o) {
   }
   size_t local_id;
   size_t global_id;
   size_t owner_id;
};
bool verifyParitions(std::string & basename, OfflineGraph & g, size_t num_hosts) {
   bool verified = true;
   std::vector<OfflineGraph*> pGraphs(num_hosts);
   std::vector<std::map<size_t, NodeInfo>> hostLocalToGlobalMap(num_hosts);
   std::cout << "Verifying partitions...\n";
   for (int h = 0; h < num_hosts; ++h) {
//      char meta_file_name[256];
//      sprintf(meta_file_name, "partition_%d_of_%zu.gr.meta", h, num_hosts);
      std::string meta_file_name = getMetaFileName(basename, h, num_hosts);
      std::ifstream meta_file(meta_file_name, std::ifstream::binary);
      if (!meta_file.is_open()) {
         std::cout << "Unable to open file " << meta_file_name << "! Exiting!\n";
         return false;
      }
      size_t num_entries;
      meta_file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
//         meta_file>>num_entries;
      std::cout << "Partition :: " << h << " Number of nodes :: " << num_entries << "\n";
      for (size_t i = 0; i < num_entries; ++i) {
         std::pair<size_t, size_t> entry;
         size_t owner;
         meta_file.read(reinterpret_cast<char*>(&entry.first), sizeof(entry.first));
         meta_file.read(reinterpret_cast<char*>(&entry.second), sizeof(entry.second));
         meta_file.read(reinterpret_cast<char*>(&owner), sizeof(owner));
         hostLocalToGlobalMap[h][entry.second] = NodeInfo(entry.second, entry.first, owner);
//         std::cout << " Global :: " << entry.first << " Local:: " << entry.second << " Owner:: " << owner << "\n";
      }

//      char gFileName[256];
//      sprintf(gFileName, "partition_%d_of_%zu.gr", h, num_hosts);
      std::string gFileName = getPartitionFileName(basename, h, num_hosts);
      pGraphs[h] = new OfflineGraph(gFileName);
   }      //End for each host.

   std::vector<size_t> outEdgeCounts(g.size());
   std::vector<size_t> inEdgeCounts(g.size());
   for (int h = 0; h < num_hosts; ++h) {
      auto & graph = *pGraphs[h];
      for (auto n = graph.begin(); n != graph.end(); ++n) {
         auto src = *n;
         assert(hostLocalToGlobalMap[h].find(src) != hostLocalToGlobalMap[h].end());
         auto g_src = hostLocalToGlobalMap[h][src].global_id;
         for (auto e = graph.edge_begin(src); e != graph.edge_end(src); ++e) {
            auto dst = graph.getEdgeDst(e);
            auto g_dst = hostLocalToGlobalMap[h][dst].global_id;
            assert(hostLocalToGlobalMap[h].find(dst) != hostLocalToGlobalMap[h].end());
            outEdgeCounts[g_src]++;
            inEdgeCounts[g_dst]++;
         }
      }
   }
   for (auto n = g.begin(); n != g.end(); ++n) {
      auto src = *n;
      for (auto e = g.edge_begin(*n); e != g.edge_end(*n); ++e) {
         outEdgeCounts[src]--;
         inEdgeCounts[g.getEdgeDst(e)]--;
      }
   }

   std::cout << "Verification sizes :: In :: " << inEdgeCounts.size() << " , Out :: " << outEdgeCounts.size() << "\n";
   for (size_t i = 0; i < inEdgeCounts.size(); ++i) {
      if (inEdgeCounts[i] != 0) {
         std::cout << "Error - Node:: " << i << " inEdgeCount!=0, count=" << inEdgeCounts[i] << "\n";
         verified = false;
      };
      if (outEdgeCounts[i] != 0) {
         std::cout << "Error - Node:: " << i << " outEdgeCount!=0, count=" << outEdgeCounts[i] << "\n";
         verified = false;
      };
   }

   for (int h = 0; h < num_hosts; ++h) {
      delete pGraphs[h];
   }
   return verified;

}
/******************************************************************
 *
 *****************************************************************/
int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   Galois::Timer T_total, T_offlineGraph_init, T_hGraph_init, T_init, T_HSSSP;
   T_total.start();
   T_hGraph_init.start();
   OfflineGraph g(inputFile);
   T_hGraph_init.stop();
   VertexCutInfo vci;
   T_init.start();
   Partitioner p;
   p(inputFile, g, vci, numPartitions);
   T_init.stop();
   if(!verifyParitions(inputFile, g, numPartitions)){
      std::cout<<"Verification of partitions failed! Contact developers!\n";
   }else{
      std::cout<<"Partitions verified!\n";
   }
   std::cout << "Completed partitioning.\n";
   return 0;
}
