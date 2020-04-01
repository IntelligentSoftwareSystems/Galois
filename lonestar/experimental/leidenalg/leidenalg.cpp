/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "galois/AtomicHelpers.h"

#include <iostream>
#include <fstream>
#include <deque>
#include <type_traits>

#include <igraph.h>

#include "Lonestar/BoilerPlate.h"
#include "leidenalg.h"
#include "galois/DynamicBitset.h"

namespace cll = llvm::cl;

static const char* name = "Louvain Clustering";

static const char* desc =
    "Cluster nodes of the graph using Louvain Clustering";

static const char* url = "louvain_clustering";

enum Algo {
  coloring,
  foreach,
  delay,
  doall
};

enum Quality{
	CPM,
	Mod
};


static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::coloring, "Coloring", "Using colors to mitigate conflicts"),
                clEnumValN(Algo::foreach, "Foreach", "Using galois for_each for conflict mitigation"),
                clEnumValN(Algo::delay, "Delay", "Using galois for_each for conflict mitigation but delay the updation"),
                clEnumValN(Algo::doall, "Doall", "Using galois for_each for conflict mitigation"),
                clEnumValEnd),
    cll::init(Algo::foreach));

static cll::opt<Quality> quality(
    "quality", cll::desc("Choose an option:"),
    cll::values(clEnumValN(Quality::CPM, "CPM", "Using CPM Quality"),
                clEnumValN(Quality::Mod, "Mod", "Using mod"),
                clEnumValEnd),
    cll::init(Quality::Mod));

static cll::opt<bool> enable_VF("enable_VF",
  cll::desc("Flag to enable vertex following optimization."),
  cll::init(false));

static cll::opt<double> c_threshold("c_threshold",
  cll::desc("Threshold for modularity gain"),
  cll::init(0.01));

static cll::opt<double> threshold("threshold",
  cll::desc("Total threshold for modularity gain"),
  cll::init(0.01));

static cll::opt<uint32_t> max_iter("max_iter",
  cll::desc("Maximum number of iterations to execute"),
  cll::init(10));

static cll::opt<bool> output_CID("output_CID",
  cll::desc("Flag to enable cluster ID printing."),
  cll::init(false));

static cll::opt<std::string> output_CID_filename("output_CID_filename",
  cll::desc("File name to output cluster IDs."),
  cll::init("output_CID_filename"));

int64_t maxQualityWithoutSwaps(Graph &graph, GNode n, CommArray &c_info){

	//compute cluster local map first
	auto& n_data = graph.getData(n, flag_write_lock);
  uint64_t degree = std::distance(graph.edge_begin(n, flag_write_lock), graph.edge_end(n,  flag_write_lock));
                   
 	int64_t local_target = -1;
  std::map<uint64_t, uint64_t> cluster_local_map; // Map each neighbor's cluster to local number: Community --> Index
  std::vector<uint64_t> counter; //Number of edges to each unique cluster
  uint64_t num_unique_clusters = 1;
  uint64_t self_loop_wt = 0;

  if(degree > 0){

 		for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
    	graph.getData(graph.getEdgeDst(ii), flag_write_lock);
    }

    cluster_local_map[graph.getData(n).curr_comm_ass] = 0; // Add n's current cluster
    counter.push_back(0); //Initialize the counter to zero (no edges incident yet)

                      
    for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
    	GNode dst = graph.getEdgeDst(ii);
     	auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
                       	
			if(dst == n){
      	self_loop_wt += edge_wt; // Self loop weights is recorded
     	}
                        
			auto stored_already = cluster_local_map.find(graph.getData(dst).curr_comm_ass); // Check if it already exists
                        
			if(stored_already != cluster_local_map.end()) {
      	counter[stored_already->second] += edge_wt;
      } else {
                        
			 cluster_local_map[graph.getData(dst).curr_comm_ass] = num_unique_clusters;
       counter.push_back(edge_wt);
       num_unique_clusters++;
      }
    } // End edge loop

    // Find the max gain
  //  local_target = maxCPMQualityWithoutSwaps(cluster_local_map, counter, self_loop_wt, c_info, n_data.degree_wt, n_data.curr_comm_ass);

		switch(quality){
          case CPM:
            local_target = maxCPMQualityWithoutSwaps(cluster_local_map, counter, self_loop_wt, c_info, n_data.degree_wt, n_data.curr_comm_ass);
            break;
          case Mod:
            local_target = maxModularityWithoutSwaps(cluster_local_map, counter, self_loop_wt, c_info, n_data.degree_wt, n_data.curr_comm_ass);
            break;
          case default:
            std::abort();
        }
	} else {
  	local_target = -1;
  }

	return local_target;
}

void moveNodesFast(Graph &graph, CommArray &c_info){

	galois::InsertBag<GNode> bag_curr;
	galois::InsertBag<GNode> bag_next;

	for(auto n:graph){
		bag_curr.push(n);
		graph.getData(n).inBag = true;
	}
	
	//updating c_info
	galois::do_all(galois::iterate(graph),
		[&] (GNode n){

		graph.getData(n).curr_comm_ass = n;
		c_info[n].size = (uint64_t) 1;
		c_info[n].deree_wt = graph.getData(n).degree_wt;
		c_info[n].flatSize = graph.getData(n).flatSize;
		}, galois::steal());

	while(true){
	
		galois::do_all(galois::iterate(bag_curr),
			[&] (GNode n){
			
				//pop out node from queue;
				graph.getData(n).inBag = false;

				//create cluster local map 
				int64_t local_target= maxQualityWithoutSwaps(graph, n, c_info);
				
				auto& n_data = graph.getData(n, flag_write_lock);
				if(local_target != -1 && local_target != graph.getData(n).curr_comm_ass){
	
					galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
          galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
					galois::atomicAdd(c_info[local_target].flatSize, n_data.flatSize);

          galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
          galois::atomicSubtract(c_info[n_data.curr_comm_ass].size, (uint64_t)1);
					galois::atomicSubtract(c_info[n_data.curr_comm_ass].flatSize, n_data.flatSize);

					graph.getData(n).curr_comm_ass = local_target;

					//explore neighbors and add them to queue
					for(auto e: graph.edges(n)){
			
						GNode u = graph.getEdgeDst(e);
						if(graph.getData(u).curr_comm_ass != local_target && !graph.getData(u).inBag){
							graph.getData(u).inBag = true;
							bag_next.push(u);
						}
					}
				}


			}, galois::steal());
	
		//breaking criterion
		if(bag_next.begin() == bag_next.end())
			break;

		//clear the queue
		bag_curr.clear();

		//fill the queue
		for(auto n: bag_next)
			bag_curr.push(n);

		bag_next.clear();
	}//end while
}

//implements SingletonPartition function
//assigns every vertex to its own subcommunity
void singletonPartition(Graph &graph, CommArray &subcomm_info){

	//call sumvertexdegreeweight if vertices'degree wt is not calculated yet
	sumVertexDegreeWeight(graph, subcomm_info);

	galois::do_all(galois::iterate(graph),
		[&] (GNode n){
		
			graph.getData(n).curr_subcomm_ass = n;
			subcomm_info[n].size = (uint64_t) 1;
			subcomm_info[n].degree_wt = graph.getData(n).degree_wt;

		}, galois::steal());
}

//this should be implemented in serial; since this is called in parallel for all communities
void mergeNodesSubset(Graph &graph, std::vector<GNode> &S, int64_t comm_id, uint64_t comm_flatSize){

	CommArray subcomm_info;

	//select set R
	std::vector<GNode> R;
	
	for(auto n: S){
		
		uint64_t total = 0;
		for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        
			GNode dst = graph.getEdgeDst(ii);
      auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
      if(dst != n && graph.getData(dst).curr_comm_ass == comm_id){
      	total += edge_wt;
			}
		}

		double flatSize_n = (double)graph.getData(n).flatSize;
		if(total >= resolution*flatSize_n*((double)comm_flatSize - flatSize_n))
			R.push_back(n);	

		subcomm_info[n].flatSize = flatSize_n;
		subcomm_info[n].external_edge_wt = total;
		subcomm_info[n].size = (uint64_t) 1;
	}

	for(auto n:R){
	
		if(subcomm_info[graph.getData(n).curr_subcomm_ass].size == (uint64_t) 1){
		
			int subcomm_ass = getRandomSubcommunity(graph, n, subcomm_info, comm_flatSize);

			graph.getData(n).curr_subcomm_ass = subcomm_ass;

			//update Subcomm info
			subcomm_info[subcomm_ass].flatSize += graph.getData(n).flatSize;
			subcomm_info[subcomm_ass].size += (uint64_t) 1;
	
			for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {

      	GNode dst = graph.getEdgeDst(ii);
      	auto edge_wt = graph.getEdgeData(ii, flag_no_lock);	
				
				if(dst != n && graph.getData(dst).curr_subcomm_ass == subcomm_ass){
					subcomm_info[subcomm_ass].external_edge_wt -= edge_wt;	
				}
				else if (dst != n){
					subcomm_info[subcomm_ass].external_edge_wt += edge_wt;
				}
			}	
		}
	}
}

void refinePartition(Graph &graph){

	//set singleton subcommunities
	galois::do_all(galois::iterate(graph),
		[&] (GNode n){
		
			graph.getData(n).curr_subcomm_ass = n;
		}, galois::steal());

	
/*	std::vector<std::set<int64_t>> myVec(101);	//to identify total no. of different communities
	std::vector<galois::InsertBag<GNode>> bags(101);	//to sort nodes according to their community

	galois::do_all(galois::iterate(graph),
		[&] (GNode n){
			
			int64_t comm_idx = graph.getData(n).curr_comm_ass%100;
			myVec[comm_idx].insert(graph.getData(n).curr_comm_ass);
			bag[comm_idx].push(n);
		}, galois::steal());
*/

	//populate nodes into communities
	std::vector<std::vector<GNode>> myVec(graph.size()+1);
	CommArray comm_info;

	comm_info.allocate(graph.size()+1);

	for(auto n: graph){

		myVec[graph.getData(n).curr_comm_ass].push_back(n);
		comm_info[graph.getData(n).curr_comm_ass].flatSize += graph.getData(n).flatSize;
	}
	
	//call mergeNodesSubset for each community in parallel	
	galois::do_all(galois::iterate((uint32_t)0, graph.size()),
  	[&](uint32_t c){
		
			if(myVec[c].size() > 0){
				mergeNodesSubset(graph, myVec[c], c, comm_info[c].flatSize);
			}
		});

	comm_info.destroy();
	comm_info.deallocate();

}



//double algoLouvainWithLocking(Graph &graph, largeArray& clusters, double lower, double threshold) {
double algoLouvainWithLocking(Graph &graph, double lower, double threshold, uint32_t& iter) {

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  galois::gPrint("Inside algoLouvainWithLocking\n");

  CommArray c_info; // Community info
  CommArray c_update; // Used for updating community
  //std::vector<Comm> c_info(graph.size()), c_update(graph.size());

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double prev_mod = lower;
  double curr_mod = -1;
  double threshold_mod = threshold;
  uint32_t num_iter = iter;


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph),
                [&graph](GNode n) {
                  graph.getData(n).curr_comm_ass = n;
                  graph.getData(n).prev_comm_ass = n;
                  graph.getData(n).colorId =  -1;
                  });

  galois::gPrint("Init Done\n");
  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[0] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

  galois::gPrint("========================================================================================================\n");
  galois::gPrint("Itr      Explore_xx            A_x2          Prev-Prev-Mod         Prev-Mod           Curr-Mod\n");
  galois::gPrint("========================================================================================================\n");

  galois::StatTimer TimerClusteringWhile("Timer_Clustering_While");
  TimerClusteringWhile.start();
  while(true) {
    num_iter++;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  c_update[n].degree_wt = 0;
                  c_update[n].size = 0;
                  });

  galois::for_each(galois::iterate(graph),
                [&](GNode n, auto& ctx) {

                    auto& n_data = graph.getData(n, flag_write_lock);
                    uint64_t degree = std::distance(graph.edge_begin(n, flag_write_lock),
                                                     graph.edge_end(n,  flag_write_lock));
                    //TODO: Can we make it infinity??
                    int64_t local_target = -1;
                    std::map<uint64_t, uint64_t> cluster_local_map; // Map each neighbor's cluster to local number: Community --> Index
                    std::vector<uint64_t> counter; //Number of edges to each unique cluster
                    uint64_t num_unique_clusters = 1;
                    uint64_t self_loop_wt = 0;

                    if(degree > 0){

                      //TODO: Make this cautious operator; Find better way
                      //Grab lock on all the neighbors before making any changes
                      for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        graph.getData(graph.getEdgeDst(ii), flag_write_lock);
                      }

                      cluster_local_map[graph.getData(n).curr_comm_ass] = 0; // Add n's current cluster
                      counter.push_back(0); //Initialize the counter to zero (no edges incident yet)

                      //Assuming we have grabbed lock on all the neighbors
                      for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        GNode dst = graph.getEdgeDst(ii);
                        auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
                        if(dst == n){
                          self_loop_wt += edge_wt; // Self loop weights is recorded
                        }
                        auto stored_already = cluster_local_map.find(graph.getData(dst).curr_comm_ass); // Check if it already exists
                        if(stored_already != cluster_local_map.end()) {
                          counter[stored_already->second] += edge_wt;
                        } else {
                         cluster_local_map[graph.getData(dst).curr_comm_ass] = num_unique_clusters;
                         counter.push_back(edge_wt);
                         num_unique_clusters++;
                        }
                      } // End edge loop

                    // Find the max gain in modularity
                    local_target = maxModularity(cluster_local_map, counter, self_loop_wt, c_info, n_data.degree_wt, n_data.curr_comm_ass, constant_for_second_term);

                    } else {
                      local_target = -1;
                    }

                    /* Update cluster info */
                    if(local_target != n_data.curr_comm_ass && local_target != -1) {

                      galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
                      galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
                      galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
                      galois::atomicSubtract(c_info[n_data.curr_comm_ass].size, (uint64_t)1);

                      //galois::atomicAdd(c_update[local_target].degree_wt, n_data.degree_wt);
                      //galois::atomicAdd(c_update[local_target].size, (uint64_t)1);
                      //galois::atomicSubtract(c_update[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
                      //galois::atomicSubtract(c_update[n_data.curr_comm_ass].size, (uint64_t)1);

                      /* Set the new cluster id */
                      n_data.curr_comm_ass = local_target;
                    }


                    /* Update c info: Asynchronously TODO: Is this correct? */
                    //galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                    //galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                }, galois::loopname("louvain algo: Phase 1"),
                   galois::no_pushes()
                );

#if 0
          galois::do_all(galois::iterate(graph),
                        [&](GNode n) {
                          galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                          galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                        });
#endif


     /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;
#if 0
    galois::GAccumulator<double> acc_e_xx;
    galois::GAccumulator<double> acc_a2_x;

    galois::do_all(galois::iterate(graph),
                  [&](GNode n) {
                    cluster_wt_internal[n] = 0;
                  });

    galois::do_all(galois::iterate(graph),
                  [&](GNode n) {
                    auto n_data = graph.getData(n);
                    for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                      if(graph.getData(graph.getEdgeDst(ii)).curr_comm_ass == n_data.curr_comm_ass) {
                        cluster_wt_internal[n] += graph.getEdgeData(ii);
                      }
                    }
                  });

    galois::do_all(galois::iterate(graph),
                  [&](GNode n) {
                    acc_e_xx += cluster_wt_internal[n];
                    acc_a2_x += (c_info[n].degree_wt) * (c_info[n].degree_wt);
                  });


    e_xx = acc_e_xx.reduce();
    a2_x = acc_a2_x.reduce();
    //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
    curr_mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term * (double)constant_for_second_term;
#endif
    curr_mod = calModularity(graph, c_info, e_xx, a2_x, constant_for_second_term);

    galois::gPrint(num_iter, "        ", e_xx, "        ", a2_x, "        ", lower, "      ", prev_mod, "       ", curr_mod, "\n");

    if((curr_mod - prev_mod) < threshold_mod){
      galois::gPrint("Modularity gain: ", (curr_mod - prev_mod) , " < ", threshold_mod, " \n");
      prev_mod = curr_mod;
      break;
    }

    prev_mod = curr_mod;

  }// End while
  TimerClusteringWhile.stop();

  iter = num_iter;

  c_info.destroy();
  c_info.deallocate();

  c_update.destroy();
  c_update.deallocate();


  TimerClusteringTotal.stop();
  return prev_mod;
}

double algoLouvainWithoutLockingDoAll(Graph &graph, double lower, double threshold, uint32_t& iter) {

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  galois::gPrint("Inside algoLouvainWithLocking\n");

  CommArray c_info; // Community info
  CommArray c_update; // Used for updating community
  //std::vector<Comm> c_info(graph.size()), c_update(graph.size());

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double prev_mod = lower;
  double curr_mod = -1;
  double threshold_mod = threshold;
  uint32_t num_iter = iter;


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph),
                [&graph](GNode n) {
                  graph.getData(n).curr_comm_ass = n;
                  graph.getData(n).prev_comm_ass = n;
                  graph.getData(n).colorId =  -1;
                  });

  galois::gPrint("Init Done\n");
  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[0] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

  galois::gPrint("========================================================================================================\n");
  galois::gPrint("Itr      Explore_xx            A_x2          Prev-Prev-Mod         Prev-Mod           Curr-Mod\n");
  galois::gPrint("========================================================================================================\n");

  galois::StatTimer TimerClusteringWhile("Timer_Clustering_While");
  TimerClusteringWhile.start();
  while(true) {
    num_iter++;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  c_update[n].degree_wt = 0;
                  c_update[n].size = 0;
                  });


  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                    auto& n_data = graph.getData(n, flag_write_lock);
                    uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock),
                                                     graph.edge_end(n,  flag_no_lock));
                    //TODO: Can we make it infinity??
                    int64_t local_target = -1;
                    std::map<uint64_t, uint64_t> cluster_local_map; // Map each neighbor's cluster to local number: Community --> Index
                    std::vector<uint64_t> counter; //Number of edges to each unique cluster
                    uint64_t num_unique_clusters = 1;
                    uint64_t self_loop_wt = 0;

                    if(degree > 0){

#if 0
                      //TODO: Make this cautious operator; Find better way
                      //Grab lock on all the neighbors before making any changes
                      for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        graph.getData(graph.getEdgeDst(ii), flag_write_lock);
                      }
#endif

                      cluster_local_map[graph.getData(n).curr_comm_ass] = 0; // Add n's current cluster
                      counter.push_back(0); //Initialize the counter to zero (no edges incident yet)

                      //Assuming we have grabbed lock on all the neighbors
                      for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        GNode dst = graph.getEdgeDst(ii);
                        auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
                        if(dst == n){
                          self_loop_wt += edge_wt; // Self loop weights is recorded
                        }
                        auto stored_already = cluster_local_map.find(graph.getData(dst).curr_comm_ass); // Check if it already exists
                        if(stored_already != cluster_local_map.end()) {
                          counter[stored_already->second] += edge_wt;
                        } else {
                         cluster_local_map[graph.getData(dst).curr_comm_ass] = num_unique_clusters;
                         counter.push_back(edge_wt);
                         num_unique_clusters++;
                        }
                      } // End edge loop

                    // Find the max gain in modularity
                    local_target = maxModularityWithoutSwaps(cluster_local_map, counter, self_loop_wt, c_info, n_data.degree_wt, n_data.curr_comm_ass, constant_for_second_term);

                    } else {
                      local_target = -1;
                    }

                    /* Update cluster info */
                    if(local_target != n_data.curr_comm_ass && local_target != -1) {

                      galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
                      galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
                      galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
                      galois::atomicSubtract(c_info[n_data.curr_comm_ass].size, (uint64_t)1);

                      //galois::atomicAdd(c_update[local_target].degree_wt, n_data.degree_wt);
                      //galois::atomicAdd(c_update[local_target].size, (uint64_t)1);
                      //galois::atomicSubtract(c_update[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
                      //galois::atomicSubtract(c_update[n_data.curr_comm_ass].size, (uint64_t)1);

                      /* Set the new cluster id */
                      n_data.curr_comm_ass = local_target;
                    }


                    /* Update c info: Asynchronously TODO: Is this correct? */
                    //galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                    //galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                }, galois::loopname("louvain algo: Phase 1")
                );

#if 0
          galois::do_all(galois::iterate(graph),
                        [&](GNode n) {
                          galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                          galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                        });
#endif


     /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;

    curr_mod = calModularity(graph, c_info, e_xx, a2_x, constant_for_second_term);

    galois::gPrint(num_iter, "        ", e_xx, "        ", a2_x, "        ", lower, "      ", prev_mod, "       ", curr_mod, "\n");

    if((curr_mod - prev_mod) < threshold_mod){
      galois::gPrint("Modularity gain: ", (curr_mod - prev_mod) , " < ", threshold_mod, " \n");
      prev_mod = curr_mod;
      break;
    }

    prev_mod = curr_mod;

  }// End while
  TimerClusteringWhile.stop();

  iter = num_iter;

  c_info.destroy();
  c_info.deallocate();

  c_update.destroy();
  c_update.deallocate();


  TimerClusteringTotal.stop();
  return prev_mod;
}




double algoLouvainWithLockingDelayUpdate(Graph &graph, double lower, double threshold, uint32_t& iter) {
  galois::gPrint("Inside algoLouvainWithLockingDelay\n");

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  CommArray c_info; // Community info
  CommArray c_update; // Used for updating community
  //std::vector<Comm> c_info(graph.size()), c_update(graph.size());

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double prev_mod = -1; //lower;
  double curr_mod = -1;
  double threshold_mod = threshold;
  uint32_t num_iter = iter;

  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph),
                [&graph](GNode n) {
                  graph.getData(n).curr_comm_ass = n;
                  graph.getData(n).prev_comm_ass = n;
                  graph.getData(n).colorId =  -1;
                  });

  galois::gPrint("Init Done\n");
  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[5] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

  galois::gPrint("========================================================================================================\n");
  galois::gPrint("Itr      Explore_xx            A_x2          Prev-Prev-Mod         Prev-Mod           Curr-Mod\n");
  galois::gPrint("========================================================================================================\n");

  galois::StatTimer TimerClusteringWhile("Timer_Clustering_While");
  TimerClusteringWhile.start();
  while(true) {
    num_iter++;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  c_update[n].degree_wt = 0;
                  c_update[n].size = 0;
                  });

  std::vector<GNode> local_target(graph.size(), -1);
  galois::GAccumulator<uint32_t> syncRound;
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {

                    auto& n_data = graph.getData(n, flag_write_lock);
                    uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock),
                                                     graph.edge_end(n,  flag_no_lock));
                    //TODO: Can we make it infinity??
                    std::map<uint64_t, uint64_t> cluster_local_map; // Map each neighbor's cluster to local number: Community --> Index
                    std::vector<uint64_t> counter; //Number of edges to each unique cluster
                    uint64_t num_unique_clusters = 1;
                    uint64_t self_loop_wt = 0;

                    if(degree > 0){

#if 0
                      //TODO: Make this cautious operator; Find better way
                      //Grab lock on all the neighbors before making any changes
                      for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        graph.getData(graph.getEdgeDst(ii), flag_write_lock);
                      }
#endif

                      cluster_local_map[graph.getData(n).curr_comm_ass] = 0; // Add n's current cluster
                      counter.push_back(0); //Initialize the counter to zero (no edges incident yet)

                      //Assuming we have grabbed lock on all the neighbors
                      for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                        GNode dst = graph.getEdgeDst(ii);
                        auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
                        if(dst == n){
                          self_loop_wt += edge_wt; // Self loop weights is recorded
                        }
                        auto stored_already = cluster_local_map.find(graph.getData(dst).curr_comm_ass); // Check if it already exists
                        if(stored_already != cluster_local_map.end()) {
                          counter[stored_already->second] += edge_wt;
                        } else {
                         cluster_local_map[graph.getData(dst).curr_comm_ass] = num_unique_clusters;
                         counter.push_back(edge_wt);
                         num_unique_clusters++;
                        }
                      } // End edge loop

                    // Find the max gain in modularity
                    local_target[n] = maxModularity(cluster_local_map, counter, self_loop_wt, c_info, n_data.degree_wt, n_data.curr_comm_ass, constant_for_second_term);

                    } else {
                      local_target[n] = -1;
                    }

                    /* Update cluster info */
                    if(local_target[n] != n_data.curr_comm_ass && local_target[n] != -1) {

                      galois::atomicAdd(c_update[local_target[n]].degree_wt, n_data.degree_wt);
                      galois::atomicAdd(c_update[local_target[n]].size, (uint64_t)1);
                      galois::atomicSubtract(c_update[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
                      galois::atomicSubtract(c_update[n_data.curr_comm_ass].size, (uint64_t)1);

                      /* Set the new cluster id */
                      //n_data.curr_comm_ass = local_target;
                    }


                    /* Update c info: Asynchronously TODO: Is this correct? */
                    //galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                    //galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                }, galois::loopname("louvain algo: Phase 1")
                );

#if 0
          galois::do_all(galois::iterate(graph),
                        [&](GNode n) {
                          galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                          galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                        });
#endif


     /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;
    curr_mod = calModularityDelay(graph, c_info, c_update, e_xx, a2_x, constant_for_second_term, local_target);
    galois::gPrint(num_iter, "        ", e_xx, "        ", a2_x, "        ", lower, "      ", prev_mod, "       ", curr_mod, "\n");

    if((curr_mod - prev_mod) < threshold_mod){
      galois::gPrint("Modularity gain: ", (curr_mod - prev_mod) , " < ", threshold_mod, " \n");
      prev_mod = curr_mod;
      break;
    }

    prev_mod = curr_mod;
    if(prev_mod < lower)
      prev_mod = lower;


     galois::do_all(galois::iterate(graph),
                   [&](GNode n) {
                     auto& n_data = graph.getData(n, flag_no_lock);
                     n_data.prev_comm_ass = n_data.curr_comm_ass;
                     n_data.curr_comm_ass = local_target[n];
                     //galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                     //galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                     c_info[n].size += c_update[n].size.load();
                     c_info[n].degree_wt += c_update[n].degree_wt.load();

                     c_update[n].size = 0;
                     c_update[n].degree_wt = 0;
                   });


  }// End while
  TimerClusteringWhile.stop();

  iter = num_iter;

  c_info.destroy();
  c_info.deallocate();

  c_update.destroy();
  c_update.deallocate();

  TimerClusteringTotal.stop();
  return prev_mod;
}



uint64_t coloringDistanceOne(Graph& graph) {
  galois::for_each(galois::iterate(graph),
                  [&](GNode n, auto& ctx){
                    auto& n_data = graph.getData(n, flag_write_lock);

                    /* Grab lock on neighbours: Cautious operator */
                    for(auto ii = graph.edge_begin(n, flag_write_lock); ii != graph.edge_end(n, flag_write_lock); ++ii) {
                      auto& dst_data = graph.getData(graph.getEdgeDst(ii), flag_write_lock); //TODO: Can we use read lock?
                    }

                    int64_t max_color = -1;
                    int64_t my_color = 0;
                    //galois::DynamicBitSet isColorSet;
                    int64_t degree = std::distance(graph.edge_begin(n), graph.edge_end(n));
                    if(degree > 0) {
                      std::vector<bool> isColorSet;
                      //std::vector<uint64_t> isColorSet;
                      isColorSet.resize(degree, false);
                      for(auto ii = graph.edge_begin(n, flag_write_lock); ii != graph.edge_end(n, flag_write_lock); ++ii) {
                        auto dst = graph.getEdgeDst(ii);
                        if(dst == n)
                          continue;

                        auto& dst_data = graph.getData(dst, flag_write_lock); //TODO: Can we use read lock?
                        if(dst_data.colorId >= 0) {
                          //isColorSet.set(dst_data.colorId);
                          if(dst_data.colorId >= degree)
                            isColorSet.resize(dst_data.colorId);

                          isColorSet[dst_data.colorId] = true;
                          if((dst_data.colorId > max_color)){
                              max_color = dst_data.colorId;
                          }
                        }
                      }

                      if(max_color >= 0) {
                      /* Assign color */
                      for (;my_color <= max_color; my_color++){
                        if(isColorSet[my_color] == false) {
                          break;
                        }
                      }

                      if(my_color == max_color)
                        my_color++;
                      }
                    }

                    n_data.colorId = my_color;

                  }, galois::loopname("Coloring loop"));

    galois::gPrint("Checking for conflicts\n");
    /* Check for conflicts */
    galois::GAccumulator<uint64_t> conflicts;
    galois::do_all(galois::iterate(graph),
                  [&](GNode n ){
                    auto& n_data = graph.getData(n, flag_no_lock);
                    for(auto ii = graph.edge_begin(n, flag_write_lock); ii != graph.edge_end(n, flag_write_lock); ++ii) {
                      auto dst = graph.getEdgeDst(ii);
                      auto& dst_data = graph.getData(dst, flag_no_lock);
                      if(dst_data.colorId == n_data.colorId)
                        conflicts += 1;
                    }
                  }, galois::loopname("Coloring conflicts"));
    galois::gPrint("WARNING: Conflicts found : ", conflicts.reduce(), "\n");

    int64_t num_colors  = 0;
    for(GNode n = 0 ; n <  graph.size(); ++n) {
      int64_t color = graph.getData(n, flag_no_lock).colorId;
      if( color > num_colors)
        num_colors = color;
    }

    return num_colors;
  }


//double algoLouvainWithColoring(Graph &graph, largeArray& clusters, double lower, double threshold) {
double algoLouvainWithColoring(Graph &graph, double lower, double threshold, uint32_t& iter) {

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  galois::gPrint("Inside algoLouvainWithColoring\n");

  CommArray c_info; // Community info
  CommArray c_update; // Used for updating community
  //std::vector<Comm> c_info(graph.size()), c_update(graph.size());

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double prev_mod = lower;
  double curr_mod = -1;
  double threshold_mod = threshold;
  uint32_t num_iter = iter;


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph),
                [&graph](GNode n) {
                  graph.getData(n).curr_comm_ass = n;
                  graph.getData(n).prev_comm_ass = n;
                  graph.getData(n).colorId = -1;
                  });

  galois::gPrint("Coloring\n");
  galois::StatTimer TimerColoring("Timer_Cloring");
  TimerColoring.start();
  int64_t num_colors = coloringDistanceOne(graph);
  TimerColoring.stop();

  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[5] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

  galois::gPrint("========================================================================================================\n");
  galois::gPrint("Itr      Explore_xx            A_x2           Prev-Mod           Curr-Mod         Time-1(s)       Time-2(s)        T/Itr(s)\n");
  galois::gPrint("========================================================================================================\n");

   galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  c_update[n].degree_wt = 0;
                  c_update[n].size = 0;
                  });

  galois::StatTimer TimerClusteringWhile("Timer_Clustering_While");
  TimerClusteringWhile.start();
  while(true) {
    num_iter++;

    for(int64_t c = 0; c < num_colors; ++c) {
        //galois::gPrint("Color : ", c, "\n");
        galois::do_all(galois::iterate(graph),
                  [&](GNode n) {

                      auto& n_data = graph.getData(n, flag_write_lock);
                      if(n_data.colorId == c){
                        uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock),
                                                         graph.edge_end(n,  flag_no_lock));
                        //TODO: Can we make it infinity??
                        int64_t local_target = -1;
                        std::map<uint64_t, uint64_t> cluster_local_map; // Map each neighbor's cluster to local number: Community --> Index
                        std::vector<uint64_t> counter; //Number of edges to each unique cluster
                        uint64_t num_unique_clusters = 1;
                        uint64_t self_loop_wt = 0;

                        if(degree > 0){
#if 0
                          //TODO: Make this cautious operator; Find better way
                          //Grab lock on all the neighbors before making any changes
                          for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                            graph.getData(graph.getEdgeDst(ii), flag_write_lock);
                          }
#endif

                          cluster_local_map[graph.getData(n).curr_comm_ass] = 0; // Add n's current cluster
                          counter.push_back(0); //Initialize the counter to zero (no edges incident yet)

                          //Assuming we have grabbed lock on all the neighbors
                          for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                            GNode dst = graph.getEdgeDst(ii);
                            auto edge_wt = graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
                            if(dst == n){
                              self_loop_wt += edge_wt; // Self loop weights is recorded
                            }
                            auto stored_already = cluster_local_map.find(graph.getData(dst).curr_comm_ass); // Check if it already exists
                            if(stored_already != cluster_local_map.end()) {
                              counter[stored_already->second] += edge_wt;
                            } else {
                             cluster_local_map[graph.getData(dst).curr_comm_ass] = num_unique_clusters;
                             counter.push_back(edge_wt);
                             num_unique_clusters++;
                            }
                          } // End edge loop

                        // Find the max gain in modularity
                        local_target = maxModularity(cluster_local_map, counter, self_loop_wt, c_info, n_data.degree_wt, n_data.curr_comm_ass, constant_for_second_term);

                        } else {
                          local_target = -1;
                        }

                        /* Update cluster info */
                        if(local_target != n_data.curr_comm_ass && local_target != -1) {

                          //galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
                          //galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
                          //galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
                          //galois::atomicSubtract(c_info[n_data.curr_comm_ass].size, (uint64_t)1);

                          galois::atomicAdd(c_update[local_target].degree_wt, n_data.degree_wt);
                          galois::atomicAdd(c_update[local_target].size, (uint64_t)1);
                          galois::atomicSubtract(c_update[n_data.curr_comm_ass].degree_wt, n_data.degree_wt);
                          galois::atomicSubtract(c_update[n_data.curr_comm_ass].size, (uint64_t)1);

                          /* Set the new cluster id */
                          n_data.curr_comm_ass = local_target;
                        }
                      }

                      /* Update c info: Asynchronously TODO: Is this correct? */
                      //galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                      //galois::atomicAdd(c_info[n].degree_wt, c_info[n].degree_wt.load());
                  }, galois::loopname("louvain algo: Phase 1")
                  );

          galois::do_all(galois::iterate(graph),
                                [&](GNode n) {
                                  galois::atomicAdd(c_info[n].size,  c_update[n].size.load());
                                  galois::atomicAdd(c_info[n].degree_wt, c_update[n].degree_wt.load());
                                  //c_info[n].size += c_update[n].size.load();
                                  //c_info[n].degree_wt += c_update[n].degree_wt.load();

                                  c_update[n].size = 0;
                                  c_update[n].degree_wt = 0;
                                });

  }

     /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;
    curr_mod = calModularity(graph, c_info, e_xx, a2_x, constant_for_second_term);

    galois::gPrint(num_iter, "        ", e_xx, "        ", a2_x, "        ", prev_mod, "       ", curr_mod, "\n");

    if((curr_mod - prev_mod) < threshold_mod){
      galois::gPrint("Modularity gain: ", (curr_mod - prev_mod) , " < ", threshold_mod, " \n");
      prev_mod = curr_mod;
      break;
    }

    prev_mod = curr_mod;

  }// End while
  TimerClusteringWhile.stop();

  iter = num_iter;

  c_info.destroy();
  c_info.deallocate();

  c_update.destroy();
  c_update.deallocate();

  TimerClusteringTotal.stop();
  return prev_mod;
}

//num_unique_clusters must be the total number of subcommunities
void buildNextLevelGraph(Graph& graph, Graph& graph_next, uint64_t num_unique_clusters) {
  std::cerr << "Inside buildNextLevelGraph\n";

  galois::StatTimer TimerGraphBuild("Timer_Graph_build");
  TimerGraphBuild.start();
  uint32_t num_nodes_next = num_unique_clusters;
  uint64_t num_edges_next = 0; //Unknown right now


  //galois::LargeArray<galois::InsertBag<uint64_t>> cluster_bags;
  //cluster_bags.allocateBlocked(num_unique_clusters);
  //std::vector<galois::InsertBag<uint64_t>> cluster_bags(num_unique_clusters);
  std::vector<std::vector<GNode>> cluster_bags(num_unique_clusters);
#if 0
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                    auto n_data = graph.getData(n, flag_no_lock);
                    cluster_bags[n_data.curr_comm_ass].push_back(n);
                },
                galois::loopname("Cluster Bags"),
                galois::steal());
#endif

	std::vector<uint64_t> cluster_flatsize(num_unique_clusters);

	//initialize to 0
	for(int i=0;i<num_unique_clusters;i++)
		cluster_flatsize = (uint64_t) 0;
	
	
  // Comment: Serial separation is better than do_all due to contention
  for(GNode n = 0; n < graph.size(); ++n) {
      auto n_data = graph.getData(n, flag_no_lock);
      if(n_data.curr_subcomm_ass != -1){
        cluster_bags[n_data.curr_subcomm_ass].push_back(n);
				cluster_flatsize[n_data.curr_subcomm_ass] += n_data.flatSize;
			}
      //else galois::gPrint("ISOLATED NODE : ", n, "\n");
  }


  std::vector<std::vector<uint32_t>> edges_id(num_unique_clusters);
  std::vector<std::vector<EdgeTy>> edges_data(num_unique_clusters);

  /* First pass to find the number of edges */
  galois::do_all(galois::iterate((uint64_t)0, num_unique_clusters),
                [&](uint64_t c) {
                    std::map<uint64_t, uint64_t> cluster_local_map;
                    uint64_t num_unique_clusters = 0;
                    for(auto cb_ii = cluster_bags[c].begin(); cb_ii != cluster_bags[c].end(); ++cb_ii) {

                      assert(graph.getData(*cb_ii, flag_no_lock).curr_subcomm_ass == c); // All nodes in this bag must have same cluster id

                      for(auto ii = graph.edge_begin(*cb_ii); ii != graph.edge_end(*cb_ii); ++ii) {
                        GNode dst = graph.getEdgeDst(ii);
                        auto dst_data = graph.getData(dst, flag_no_lock);
                        //assert(dst_data.curr_comm_ass < INF_VAL);
                        assert(dst_data.curr_subcomm_ass !=  -1);
                        auto stored_already = cluster_local_map.find(dst_data.curr_subcomm_ass); // Check if it already exists
                        if(stored_already != cluster_local_map.end()) {
                          edges_data[c][stored_already->second] += graph.getEdgeData(ii);
                        } else {
                          cluster_local_map[dst_data.curr_subcomm_ass] = num_unique_clusters;
                          edges_id[c].push_back(dst_data.curr_subcomm_ass);
                          edges_data[c].push_back(graph.getEdgeData(ii));
                          num_unique_clusters++;
                        }
                      } // End edge loop
                    }
                }, galois::steal(),
                   galois::loopname("BuildGrah: Find edges"));

  /* Serial loop to reduce all the edge counts */
  std::vector<uint64_t> prefix_edges_count(num_unique_clusters);
  galois::GAccumulator<uint64_t> num_edges_acc;
  galois::do_all(galois::iterate((uint32_t)0, num_nodes_next),
                [&](uint32_t c){
                  prefix_edges_count[c] = edges_id[c].size();
                  num_edges_acc += prefix_edges_count[c];
                });

  num_edges_next = num_edges_acc.reduce();
  for(uint32_t c = 1; c < num_nodes_next; ++c) {
    prefix_edges_count[c] += prefix_edges_count[c - 1];
  }

  assert(prefix_edges_count[num_unique_clusters - 1] == num_edges_next);
  galois::gPrint("#nodes : ", num_nodes_next, ", #edges : ", num_edges_next, "\n");
  galois::gPrint("#prefix last : ", prefix_edges_count[num_unique_clusters - 1], "\n");

#if 0
  for(uint32_t i = 0; i < num_nodes_next; ++i){
    for(uint32_t j = 0; j < edges_id[i].size(); ++j){
      galois::gPrint(i, " -B-> <", edges_id[i][j], " , ", edges_data[i][j], ">\n");
    }
  }
#endif

  std::cerr << "Graph construction started" << "\n";

  galois::StatTimer TimerConstructFrom("Timer_Construct_From");
  TimerConstructFrom.start();
  graph_next.constructFrom(num_nodes_next, num_edges_next, prefix_edges_count, edges_id, edges_data);
  TimerConstructFrom.stop();

	//setting flatSizes
	galois::do_all(galois::iterate(graph_next),
		[&] (GNode n){
			
			graph_next.getData(n).flatSize = cluster_flatsize[n];
		}, galois::steal());

  TimerGraphBuild.stop();
  galois::gPrint("Graph construction done\n");
}
void runMultiPhaseLouvainAlgorithm(Graph& graph, uint64_t min_graph_size, double c_threshold, largeArray& clusters_orig) {

  galois::gPrint("Inside runMultiPhaseLouvainAlgorithm\n");
  double prev_mod = -1; //Previous modularity
  double curr_mod = -1; //Current modularity
  uint32_t phase = 0;

  Graph* graph_curr = &graph;
  Graph graph_next;
  uint32_t iter = 0;
  bool non_color = false;
  uint64_t num_nodes_orig = clusters_orig.size();
  while(true){
    iter++;
    phase++;
    galois::gPrint("Starting Phase : ", phase, "\n");
    galois::gPrint("Graph size : ", (*graph_curr).size(), "\n");

    //TODO: add to the if conditional
    if((*graph_curr).size() > min_graph_size){

        switch (algo) {
          case coloring:
                curr_mod = algoLouvainWithColoring(*graph_curr, curr_mod, c_threshold, iter);
                break;
          case foreach:
                curr_mod = algoLouvainWithLocking(*graph_curr, curr_mod, c_threshold, iter);
                break;
          case doall:
                curr_mod = algoLouvainWithoutLockingDoAll(*graph_curr, curr_mod, c_threshold, iter);
                break;
          case delay:
                curr_mod = algoLouvainWithLockingDelayUpdate(*graph_curr, curr_mod, c_threshold, iter);
                break;
          default:
                std::abort();
        }
    }

    uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
    galois::gPrint("Number of unique clusters (renumber): ", num_unique_clusters, "\n");

    if(!enable_VF && phase == 1){
      assert(num_nodes_orig == (*graph_curr).size());
      galois::do_all(galois::iterate(*graph_curr),
                    [&](GNode n) {
                      clusters_orig[n] = (*graph_curr).getData(n, flag_no_lock).curr_comm_ass;
                    });
    } else {
      galois::do_all(galois::iterate((uint64_t)0, num_nodes_orig),
                    [&](GNode n) {
                      if(clusters_orig[n] > 0){
                        assert(clusters_orig[n] < (*graph_curr).size());
                        //galois::gPrint(clusters_orig[n],"\n");
                        clusters_orig[n] = (*graph_curr).getData(clusters_orig[n], flag_no_lock).curr_comm_ass;
                      }
                    });

      //galois::gPrint("ARRAY unique cluster: ", renumberClustersContiguouslyArray(clusters_orig), "\n");
    }
    galois::gPrint("Prev_mod main: ", prev_mod, "\n");
    if(iter < max_iter && (curr_mod - prev_mod) > threshold) {
      buildNextLevelGraph(*graph_curr, graph_next, num_unique_clusters);
      prev_mod = curr_mod;
      graph_curr = &graph_next;
      printGraphCharateristics(*graph_curr);
    } else {
      break;
    }
  }
#if 0
  if(!non_color && algo != delay) {
    iter++;
    galois::gPrint("Executing one non-foreach round\n");
    curr_mod = algoLouvainWithLockingDelayUpdate(*graph_curr, prev_mod, c_threshold, iter); //Run at least one non-foreach loop
    non_color = true;
    uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
    galois::gPrint("Number of unique clusters (renumber): ", num_unique_clusters, "\n");
  }
#endif
  galois::gPrint("Phases : ", phase, "\n");
  galois::gPrint("Iter : ", iter, "\n");
}

void leiden(Graph &graph){


	Graph* graph_curr;
	Graph graph_next;

	graph_curr = &graph;	
	while(true){

		CommArray c_info;
		c_info.allocate(graph_curr->size()+1);
	
		moveNodesFast(*graph_curr, c_info);

		bool done = true;

		//check if done or not
		galois::do_all(galois::iterate(*graph_curr),
			[&] (GNode n){

				if(c_info[n].size > 1)
					done = false;
			});

		//termination criterion
		if(done)
			break;

		refinePartition(*graph_curr);

		uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
		buildNextLevelGraph(*graph_curr, graph_next, num_unique_clusters);

		graph_curr = &graph_next;					

		//destroying c_info
		c_info.destroy();
		c_info.deallocate();	
	}
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph, graph_next;
  Graph *graph_curr;

  galois::StatTimer TEnd2End("Timer_end2end");
  TEnd2End.start();

	set_rng();

  std::cout << "Reading from file: " << filename << std::endl;
  std::cout << "[WARNING:] Make sure " << filename << " is symmetric graph without duplicate edges" << std::endl;
  galois::graphs::readGraph(graph, filename);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges" << std::endl;

  graph_curr = &graph;

  /*
   * To keep track of communities for nodes in the original graph.
   *Community will be set to -1 for isolated nodes
   */
  largeArray clusters_orig;
  clusters_orig.allocateBlocked(graph_curr->size());

	//initialize flat sizes to 1
	galois::do_all(galois::iterate(*graph_curr),
                  [&](GNode n){	

										graph.getData(n).flatSize = (uint64_t) 1;									
									},galois::steal());
  /*
   * Vertex following optimization
   */
  if (enable_VF){
    uint64_t num_nodes_to_fix = vertexFollowing(graph); // Find nodes that follow other nodes
    galois::gPrint("Isolated nodes : ", num_nodes_to_fix, "\n");

    uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
    galois::gPrint("Number of unique clusters (renumber): ", num_unique_clusters, "\n");
    /*
     *Initialize node cluster id.
     */
    galois::do_all(galois::iterate(*graph_curr),
                  [&](GNode n){
                    clusters_orig[n] = graph.getData(n, flag_no_lock).curr_comm_ass;
                  });

    //Build new graph to remove the isolated nodes
    buildNextLevelGraph(*graph_curr, graph_next, num_unique_clusters);
    graph_curr = &graph_next;
    printGraphCharateristics(*graph_curr);
  } else {

    /*
     *Initialize node cluster id.
     */
    galois::do_all(galois::iterate(*graph_curr),
                  [&](GNode n){
                    clusters_orig[n] = -1;
                  });

    printGraphCharateristics(*graph_curr);
  }

  uint64_t min_graph_size = 10;
  //runMultiPhaseLouvainAlgorithm(graph, clusters_orig, min_graph_size, c_threshold);
  galois::gPrint("GOING in \n");
  galois::StatTimer Tmain("Timer_LC");
  Tmain.start();
  runMultiPhaseLouvainAlgorithm(*graph_curr, min_graph_size, c_threshold, clusters_orig);
  Tmain.stop();

  TEnd2End.stop();

  /*
   * Sanity check: Check modularity at the end
   */

  checkModularity(graph, clusters_orig);
  if(output_CID){
    printNodeClusterId(graph, output_CID_filename);
  }
  return 0;
}
