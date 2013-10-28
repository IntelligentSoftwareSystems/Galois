/* 
 * License:
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * Stochastic gradient descent for matrix factorization, implemented with Galois.
 *
 * Author: Prad Nelluru <pradn@cs.utexas.edu>
*/

#include <iostream>
#include <random>
#include <cmath>

#include "Galois/Galois.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"

typedef struct Node
{
  double* latent_vector; //latent vector to be learned
  unsigned int updates; //number of updates made to this node (only used by movie nodes)
  unsigned int edge_offset; //if a movie's update is interrupted, where to start when resuming
} Node;

//local computation graph (can't add nodes/edges at runtime)
//node data is Node, edge data is unsigned int
typedef Galois::Graph::LC_CSR_Graph<Node, unsigned int> Graph;

unsigned int LATENT_VECTOR_SIZE = 100;
double LEARNING_RATE = 0.01;
double DECAY_RATE = 0.1;
double LAMBDA = 1.0;
unsigned int MAX_MOVIE_UPDATES = 5;

struct sgd
{
	Graph& g;
	sgd(Graph& g) : g(g) {}
	
	//perform SGD update on all edges of a movie
	//perform update on one user at a time
	void operator()(Graph::GraphNode movie, Galois::UserContext<Graph::GraphNode>& ctx)
	{	
		Node& movie_data = g.getData(movie);
		double* movie_latent = movie_data.latent_vector;
		double step_size = LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(movie_data.updates + 1, 1.5));
		
                Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::NONE) + movie_data.edge_offset;
		Graph::GraphNode user = g.getEdgeDst(edge_it);
		//abort operation if conflict detected (Galois::ALL)
		Node& user_data = g.getData(user, Galois::ALL);
		double* user_latent = user_data.latent_vector;
		//abort operation if conflict detected (Galois::ALL)
		unsigned int edge_rating = g.getEdgeData(edge_it, Galois::ALL);	

		//calculate error
		double cur_error = - edge_rating;
		for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
		{
			cur_error += user_latent[i] * movie_latent[i];
		}

		//take gradient step
		for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
		{
			double prev_movie_val = movie_latent[i];

			movie_latent[i] -= step_size * (cur_error * user_latent[i] + LAMBDA * prev_movie_val);
			user_latent[i] -= step_size * (cur_error * prev_movie_val + LAMBDA * user_latent[i]);
		}
		
		++edge_it;
		movie_data.edge_offset++;
		//we just looked at the last user
		if(edge_it == g.edge_end(movie, Galois::NONE))
		{
			//start back at the first edge
                  movie_data.edge_offset = 0;

			//push movie node onto worklist if it's not updated enough
			movie_data.updates++;
			if(movie_data.updates < MAX_MOVIE_UPDATES)
				ctx.push(movie);
		}
		else //haven't looked at all the users this iteration
		{
			ctx.push(movie);
		}
		
	} 
};

// Initializes latent vector and id for each node
unsigned int initializeGraphData(Graph& g)
{
	unsigned int seed = 42;
	std::default_random_engine eng(seed);
	std::uniform_real_distribution<double> random_lv_value(0, 0.1);
	
	unsigned int num_movie_nodes = 0;

	//for all movie and user nodes in the graph
	for (Graph::iterator i = g.begin(), end = g.end(); i != end; ++i) {
		Graph::GraphNode gnode = *i;
		Node& data = g.getData(gnode);
		
		data.updates = 0;

		//fill latent vectors with random values
		double* lv = new double[LATENT_VECTOR_SIZE];
		for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
		{
			lv[i] = random_lv_value(eng);
		}
		data.latent_vector = lv;
		
		//count number of movies we've seen; only movies nodes have edges
		unsigned int num_edges = 
			g.edge_end(gnode, Galois::NONE) - g.edge_begin(gnode, Galois::NONE);
		if(num_edges > 0)
			num_movie_nodes++;
		
		data.edge_offset = 0;
	}

	return num_movie_nodes;
}

Graph* g_ptr;

struct projCount : public std::unary_function<unsigned, Graph::GraphNode&> {
	unsigned operator()(const Graph::GraphNode& node) const {
		return g_ptr->getData(node, Galois::NONE).updates;
	}
};

int main(int argc, char** argv) {	
	if(argc < 3)
	{
		std::cout << "Usage: <input binary gr file> <thread count>" << std::endl;
		return -1;
	}
	
	//read command line parameters
	const char* inputFile = argv[1];
	unsigned int threadCount = atoi(argv[2]);

	//how many threads Galois should use
	Galois::setActiveThreads(threadCount);

	//prints out the number of conflicts at the end of the program
	Galois::StatManager statManager;

	//allocate local computation graph
	Graph g;
	g_ptr = &g;

	//read structure of graph & edge weights; nodes not initialized
        Galois::Graph::readGraph(g, inputFile);

	//fill each node's id & initialize the latent vectors
	unsigned int num_movie_nodes = initializeGraphData(g);
	
	std::cout << "Movies, " << num_movie_nodes << ",Users, " << g.size() - num_movie_nodes <<
		",Ratings, " << g.sizeEdges() << ",Threads, " << threadCount << std::endl;

	Galois::StatTimer timer;
	timer.start();

	//do the SGD computation in parallel
	//the initial worklist contains all the movie nodes
	//the movie nodes are located at the top num_movie_nodes nodes in the graph
	//the worklist is a priority queue ordered by the number of updates done to a movie
	//the projCount functor provides the priority function on each node
	Graph::iterator ii = g.begin();
	std::advance(ii,(g.size() - num_movie_nodes)); //advance moves passed in iterator
	Galois::for_each(ii, g.end(), sgd(g),
                         Galois::wl<Galois::WorkList::OrderedByIntegerMetric
                         <projCount, Galois::WorkList::dChunkedLIFO<32>>>());

	timer.stop();

	return 0;
}
