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


#ifndef LOUVAIN_CLUSTERING_H
#define LOUVAIN_CLUSTERING_H

/*
 * Typedefs
 */

constexpr static const uint64_t INF_VAL = std::numeric_limits<uint64_t>::max() / 2 - 1;
constexpr static const double INF_VAL_DOUBLE = std::numeric_limits<double>::max() / 2 - 1;


constexpr galois::MethodFlag flag_no_lock = galois::MethodFlag::UNPROTECTED;
constexpr galois::MethodFlag flag_read_lock = galois::MethodFlag::READ;
constexpr galois::MethodFlag flag_write_lock = galois::MethodFlag::WRITE;

typedef galois::LargeArray<int64_t> largeArray;

//Maintain community information
struct Comm {
  std::atomic<uint64_t> size;
  std::atomic<uint64_t> degree_wt;
	std::atomic<uint64_t> flatSize;
	std::atomic<uint64_t> external_edge_wt;
};

typedef galois::LargeArray<Comm> CommArray;

//Graph Node information
struct Node{
  int64_t prev_comm_ass;
  int64_t curr_comm_ass;
	
	int64_t prev_subcomm_ass;
	int64_t curr_subcomm_ass;

  uint64_t degree_wt;
	uint64_t internal_degree_wt;

  //uint64_t cluster_wt_internal;
  int64_t colorId;

	uint64_t flatSize;
	uint64_t external_edge_wt;
};

typedef uint64_t EdgeTy;
using Graph =
    galois::graphs::LC_CSR_Graph<Node, EdgeTy>::with_no_lockable<false>::type::with_numa_alloc<true>::type;

using GNode = Graph::GraphNode;

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0,1.0);

double resolution;
double randomness;
double constant;

void setResolution(double res){

	resolution = res;
}

void setRandomness(double rdm){

	randomness = rdm;
}

//verbatim as is from the Java code
double fastExp(double exponent){

	if(exponent < -256.0f)
		return 0;

	exponent = 1.0f + exponent/256.0f ;
	exponent *= exponent;
	exponent *= exponent;
	exponent *= exponent;
	exponent *= exponent;
	exponent *= exponent;
	exponent *= exponent;
	exponent *= exponent;
	exponent *= exponent;
	
	return exponent;
}

void printGraphCharateristics(Graph& graph) {

  galois::gPrint("/******************************************/\n");
  galois::gPrint("/************ Graph Properties ************/\n");
  galois::gPrint("/******************************************/\n");
  galois::gPrint("Number of Nodes: ", graph.size(), "\n");
  galois::gPrint("Number of Edges: ", graph.sizeEdges(), "\n");

}


void sumVertexDegreeWeight(Graph& graph, CommArray& c_info) {
//void sumVertexDegreeWeight(Graph& graph, std::vector<Comm>& c_info) {
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  uint64_t total_weight = 0;
                  auto &n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii){
                     total_weight += graph.getEdgeData(ii, flag_no_lock);
                  }
                  n_data.degree_wt = total_weight;
                  c_info[n].degree_wt = total_weight;
                  c_info[n].size = 1;
                }, galois::steal());
}

void sumClusterWeight(Graph& graph, CommArray& c_info) {
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  uint64_t total_weight = 0;
                  auto &n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii){
                     total_weight += graph.getEdgeData(ii, flag_no_lock);
                  }
                  n_data.degree_wt = total_weight;
                  c_info[n].degree_wt = 0;
                }, galois::steal());

	galois::do_all(galois::iterate(graph),
                [&](GNode n) {
      
			auto &n_data = graph.getData(n);
      galois::atomicAdd(c_info[n_data.curr_subcomm_ass].degree_wt, n_data.degree_wt);
   }, galois::steal());
}

double calConstantForSecondTerm(Graph& graph){
	
	galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  uint64_t total_weight = 0;
                  auto &n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii){
                     total_weight += graph.getEdgeData(ii, flag_no_lock);
                  }
                  n_data.degree_wt = total_weight;
			}, galois::steal());

  galois::GAccumulator<uint64_t> local_weight;
  galois::do_all(galois::iterate(graph),
                [&graph, &local_weight](GNode n){
                  local_weight += graph.getData(n).degree_wt;
                }, galois::steal());
  /* This is twice since graph is symmetric */
  uint64_t total_edge_weight_twice = local_weight.reduce();

  return 1/(double)total_edge_weight_twice;
}

void setConstant(Graph& graph){
	constant = calConstantForSecondTerm(graph);
}

double diffModQuality(uint64_t curr_subcomm, uint64_t candidate_subcomm, std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, CommArray &subcomm_info, uint64_t self_loop_wt, uint64_t degree_wt){

  double ax = subcomm_info[curr_subcomm].degree_wt - degree_wt;
  double ay = subcomm_info[candidate_subcomm].degree_wt;

  double diff = 2.0f*(double)(counter[cluster_local_map[candidate_subcomm]] - counter[cluster_local_map[curr_subcomm]] + self_loop_wt) + 2.0f*(double)degree_wt* constant *(double)(ax - ay);

//  return diff*constant;
 return diff;
}

uint64_t maxModularity(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt.load(); // Degree wt of cluster y
      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      
			cur_gain = 2 * constant * (eiy - eix) + 2 * degree_wt * ((ax - ay) * constant * constant);
      
			if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first > max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  if((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index < sc)) {
    max_index = sc;
  }

  assert(max_gain >= 0);

  return max_index;
}

uint64_t maxModularityWithoutSwaps(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

	double size_x = c_info[sc].size;
	double size_y = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y

			size_y = c_info[stored_already->first].size;
			//if(ay < (ax + degree_wt)){
			if(size_y < size_x){
				stored_already++;	
				continue;
			}
			else if (size_y == size_x && stored_already->first < sc)
			{
				stored_already++;
				continue;
			}

      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      
			cur_gain = 2 * constant * (eiy - eix) + 2 * degree_wt * ((ax - ay) * constant * constant);

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first > max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  if((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index < sc)) {
    max_index = sc;
  }

  assert(max_gain >= 0);
  return max_index;
}


double calModularityDelay(Graph& graph, CommArray& c_info, CommArray& c_update, double& e_xx, double& a2_x, double& constant_for_second_term, std::vector<GNode>& local_target) {

  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArray cluster_wt_internal;


  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());


   /* Calculate the overall modularity */
  galois::GAccumulator<double> acc_e_xx;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(local_target[graph.getEdgeDst(ii)] == local_target[n]) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].degree_wt + c_update[n].degree_wt) * ((double) (c_info[n].degree_wt + c_update[n].degree_wt) * (double)constant_for_second_term);
                }, galois::steal());


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term;

  return mod;
}



double calModularity(Graph& graph, CommArray& c_info, double& e_xx, double& a2_x, double& constant_for_second_term) {

  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArray cluster_wt_internal;


  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());


   /* Calculate the overall modularity */
  galois::GAccumulator<double> acc_e_xx;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_comm_ass == n_data.curr_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].degree_wt) * ((double) (c_info[n].degree_wt) * (double)constant_for_second_term);
                }, galois::steal());


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term;

  return mod;
}

double calModularityFinal(Graph& graph) {
  CommArray c_info; // Community info
  CommArray c_update; // Used for updating community

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double mod = -1;

  largeArray cluster_wt_internal;
  //std::vector<uint64_t> cluster_wt_internal(graph.size());


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());
  cluster_wt_internal.allocateBlocked(graph.size());

  /* Calculate the weighted degree sum for each vertex */
  sumClusterWeight(graph, c_info);

  /* Compute the total weight (2m) and 1/2m terms */
//  constant_for_second_term = calConstantForSecondTerm(graph);
constant_for_second_term = constant;

   /* Calculate the overall modularity */
  double e_xx = 0;




  galois::GAccumulator<double> acc_e_xx;
  double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_subcomm_ass == n_data.curr_subcomm_ass) {
                    //if(graph.getData(graph.getEdgeDst(ii)).prev_comm_ass == n_data.prev_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].degree_wt.load()) * ((double) (c_info[n].degree_wt.load())* (double)constant_for_second_term);
								}, galois::steal());

	
	e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();
	

	std::cout <<"exx:" << e_xx << " a2x: "  <<a2_x << std::endl;
	mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term;
  return mod;
}

double diffCPMQuality(uint64_t curr_subcomm, uint64_t candidate_subcomm, 
std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, 
CommArray &subcomm_info, uint64_t self_loop_wt, uint64_t flatSize){

	uint64_t size_x = subcomm_info[curr_subcomm].flatSize;
	uint64_t size_y = subcomm_info[candidate_subcomm].flatSize;

	uint64_t new_size_x = size_x - flatSize;
	uint64_t new_size_y = size_y + flatSize;

	double diff1 = 2.0f*(double)(counter[cluster_local_map[candidate_subcomm]] - counter[cluster_local_map[curr_subcomm]] + self_loop_wt);
	double diff2 = resolution * 0.5f*(double)((size_x*(size_x-1) + size_y*(size_y-1)) - ((new_size_x)*(new_size_x-1) + new_size_y*(new_size_y-1)));

	double diff = diff1 + diff2;

	return diff;
//	return diff*constant;
}

uint64_t maxCPMQuality(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc, uint64_t flatSize) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
//  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  //double ay = 0;

	double size_x = c_info[sc].flatSize;
	double new_size_x = size_x - (double) flatSize;


  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      //ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y
			double size_y = c_info[stored_already->first].flatSize;
			double new_size_y = size_y + (double) flatSize;

      eiy = counter[stored_already->second]; // Total edges incident on cluster y
   		
			cur_gain = 2.0f * (double)(eiy - eix) + 0.5f*resolution*((double)(size_x*(size_x - 1) + size_y*(size_y - 1)) - (double)((new_size_x)*(new_size_x-1) + (new_size_y)*(new_size_y-1)));   

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first > max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  assert(max_gain >= 0);
  return max_index;
}

	

uint64_t maxCPMQualityWithoutSwaps(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc, uint64_t flatSize) {

  int64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
 // double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  //double ay = 0;

	double size_x = c_info[sc].flatSize;
	double size_y = 0;


	double new_size_x = size_x - (double)flatSize;
	double new_size_y = 0;	

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
    //  ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y

			size_y = c_info[stored_already->first].flatSize;
			new_size_y = size_y + (double)flatSize;

			//if(ay < (ax + degree_wt)){
			if(size_y < size_x){
				stored_already++;	
				continue;
			}
			else if (size_y == size_x && stored_already->first < sc)
			{
				stored_already++;
				continue;
			}

      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      
			cur_gain = 2.0f * (double)(eiy - eix) + 0.5f*resolution*((double)(size_x*(size_x - 1) + size_y*(size_y - 1)) - (double)((new_size_x)*(new_size_x-1) + (new_size_y)*(new_size_y-1)));

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && ((int64_t) stored_already->first > max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  assert(max_gain >= 0);
  return max_index;
}


//c_info must have updated size values
double calCPMQuality(Graph& graph, CommArray& c_info, double& e_xx, double& a2_x, double resolution) {

  /* Variables needed for CPM quality calculation */
  double mod = -1;

  largeArray cluster_wt_internal;


  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());


   /* Calculate the overall modularity */
  galois::GAccumulator<double> acc_e_xx;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_subcomm_ass == n_data.curr_subcomm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].flatSize) * ((double) (c_info[n].flatSize - 1)) * 0.5f;
                }, galois::steal());


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx - a2_x*resolution;

  return mod;
}



/*
 * To compute the final modularity using prev cluster
 * assignments.
 */
double calCPMQualityFinal(Graph& graph) {
  CommArray c_info; // Community info

  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArray cluster_wt_internal;
  //std::vector<uint64_t> cluster_wt_internal(graph.size());


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  cluster_wt_internal.allocateBlocked(graph.size());

   /* Calculate the overall modularity */
  double e_xx = 0;
  galois::GAccumulator<double> acc_e_xx;
  double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
									c_info[n].flatSize = 0;
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_subcomm_ass == n_data.curr_subcomm_ass) {
                    //if(graph.getData(graph.getEdgeDst(ii)).prev_comm_ass == n_data.prev_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }

									//updating community size
									galois::atomicAdd(c_info[n_data.curr_subcomm_ass].flatSize, n_data.flatSize);
                }, galois::steal());

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (double) (c_info[n].flatSize) * ((double) (c_info[n].flatSize - 1)* 0.5f);
								}, galois::steal());


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx - a2_x*resolution;
  return mod;
}




uint64_t renumberClustersContiguously(Graph &graph) {

  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < graph.size(); ++n){
    auto& n_data = graph.getData(n, flag_no_lock);
		assert(n_data.curr_subcomm_ass != -1);
    if(n_data.curr_subcomm_ass != -1) {
 //     assert(n_data.curr_subcomm_ass < graph.size());
      auto stored_already = cluster_local_map.find(n_data.curr_subcomm_ass);
     if(stored_already != cluster_local_map.end()){
      n_data.curr_subcomm_ass = stored_already->second;
     } else {
      cluster_local_map[n_data.curr_subcomm_ass] = num_unique_clusters;
      n_data.curr_subcomm_ass = num_unique_clusters;
      num_unique_clusters++;
     }
    }
  }

  return num_unique_clusters;
}

uint64_t renumberClustersContiguouslyArray(largeArray &arr) {

  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < arr.size(); ++n){
    if(arr[n] != -1) {
      assert(arr[n] < arr.size());
      auto stored_already = cluster_local_map.find(arr[n]);
     if(stored_already != cluster_local_map.end()){
      arr[n] = stored_already->second;
     } else {
      cluster_local_map[arr[n]] = num_unique_clusters;
      arr[n] = num_unique_clusters;
      num_unique_clusters++;
     }
    }
  }

	return num_unique_clusters;
}


void printGraph(Graph& graph){
  for(GNode n = 0; n < graph.size(); ++n) {
    for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      galois::gPrint(n, " --> ", graph.getEdgeDst(ii), " , ", graph.getEdgeData(ii), "\n");
    }
  }
}

void printNodeClusterId(Graph& graph, std::string output_CID_filename){
  std::ofstream outputFile(output_CID_filename, std::ofstream::out);
  for(GNode n = 0; n < graph.size(); ++n) {
    outputFile << n << "  " << graph.getData(n).curr_comm_ass << "\n";
  }
}

void checkModularity(Graph& graph, largeArray& clusters_orig) {
  galois::gPrint("checkModularity\n");

  galois::do_all(galois::iterate(graph),
                [&](GNode n){
                  graph.getData(n, flag_no_lock).curr_subcomm_ass = clusters_orig[n];
                }, galois::steal());

  uint64_t num_unique_clusters = renumberClustersContiguously(graph);
  galois::gPrint("Number of unique clusters (renumber): ", num_unique_clusters, "\n");
  auto mod = calModularityFinal(graph);
  galois::gPrint("FINAL MOD: ", mod, "\n");
}

void checkCPMQuality(Graph& graph, largeArray& clusters_orig) {
  galois::gPrint("checkModularity\n");

  galois::do_all(galois::iterate(graph),
                [&](GNode n){
                  graph.getData(n, flag_no_lock).curr_subcomm_ass = clusters_orig[n];
                }, galois::steal());

  uint64_t num_unique_clusters = renumberClustersContiguously(graph);
  galois::gPrint("Number of unique clusters (renumber): ", num_unique_clusters, "\n");
  auto mod = calCPMQualityFinal(graph);
  galois::gPrint("FINAL MOD: ", mod, "\n");
}


#endif //LOUVAIN_CLUSTERING_H
