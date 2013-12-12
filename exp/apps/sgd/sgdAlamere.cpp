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
#include <algorithm>


#include <cmath>
#include <cstdlib>

#include "Galois/Accumulator.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"

#include "Galois/Graph/Graph.h"
#include "Galois/Graph/LCGraph.h"

#include "Galois/Runtime/ll/PaddedLock.h"

#include "Lonestar/BoilerPlate.h"

static const char* const name = "Stochastic Gradient Descent";
static const char* const desc = "Computes Matrix Decomposition using Stochastic Gradient Descent";
static const char* const url = "sgd";

enum Algo {
	block,
	blockAndSliceUsers,
	blockAndSliceBoth,
	sliceMarch
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile (cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned> usersPerBlockSlice ("usersPerBlk", cll::desc ("[users per block slice]"), cll::init (2048));
static cll::opt<unsigned> moviesPerBlockSlice ("moviesPerBlk", cll::desc ("[movies per block slice]"), cll::init (512));

static cll::opt<Algo> algo(cll::desc("Choose an algorithm:"),
		cll::values(
			clEnumVal(block, "Block by Users and Movies"),
			clEnumVal(blockAndSliceUsers, "Block by Users and Movies, Slice by Users"),
			clEnumVal(blockAndSliceBoth, "Block by Users and Movies, Slice by Users and Movies (default)"),
			clEnumVal(sliceMarch, "Marching Slices version"),
			clEnumValEnd), 
		cll::init(blockAndSliceBoth));

struct Node {
	double* latent_vector; //latent vector to be learned
	unsigned int updates; //number of updates made to this node (only used by movie nodes)
	unsigned int edge_offset; //if a movie's update is interrupted, where to start when resuming
};

//local computation graph (can't add nodes/edges at runtime)
//node data is Node, edge data is unsigned int
// typedef Galois::Graph::LC_CSR_Graph<Node, unsigned int> Graph;

typedef typename Galois::Graph::LC_CSR_Graph<Node, unsigned>
::with_numa_alloc<true>::type
::with_no_lockable<false>::type Graph;
typedef Graph::GraphNode GNode;

struct ThreadWorkItem {
	unsigned int movieRangeStart;
	unsigned int movieRangeEnd;
	unsigned int userRangeStart;
	unsigned int userRangeEnd;
	unsigned int usersPerBlockSlice;
	unsigned int moviesPerBlockSlice;

	unsigned int sliceStart; //only used in march variation
	unsigned int numSlices; //only used in march variation

	//debug
	unsigned int id;
	unsigned int updates;
	double timeTaken;
};

unsigned int NUM_MOVIE_NODES = 0;
unsigned int NUM_USER_NODES = 0;
unsigned int NUM_RATINGS = 0;

static const unsigned int LATENT_VECTOR_SIZE = 20;
static const unsigned int MAX_MOVIE_UPDATES = 5;
static const double MINVAL = -1e+100;
static const double MAXVAL = 1e+100;

static const double LEARNING_RATE = 0.001; // GAMMA
static const double DECAY_RATE = 0.9; // STEP_DEC
static const double LAMBDA = 0.001;

void print_latent(double* vec)
{
	for(unsigned i = 0; i < LATENT_VECTOR_SIZE; i++)
	{
		printf("%f ", vec[i]);
	}
	printf("\n");
}

double calcPrediction (const Node& movie_data, const Node& user_data) {
	// XXX: __restrict__ may not be necessary here
	const double* __restrict__ movie_latent = movie_data.latent_vector;
	const double* __restrict__ user_latent = user_data.latent_vector;

	double pred = 0.0;
	for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
	{
		pred += user_latent[i] * movie_latent[i];
	}

	pred = std::min (MAXVAL, pred);
	pred = std::max (MINVAL, pred);

	return pred;
}

/*inline*/ void doGradientUpdate(Node& movie_data, Node& user_data, unsigned int edge_rating)
{
	//Purdue folks' learning function:
	//double step_size = LEARNING_RATE * 1.5 / (1.0 + DECAY_RATE * pow(movie_data.updates + 1, 1.5));
	//Intel folks' learning function:
	double step_size = LEARNING_RATE * pow (DECAY_RATE, movie_data.updates);
	
	double* __restrict__ movie_latent = movie_data.latent_vector;
	double* __restrict__ user_latent = user_data.latent_vector;

	//calculate error using calcPrediction -> leads to some slowdown
	//  double pred = calcPrediction (movie_data, user_data);
	//  double cur_error = pred - edge_rating;
	
	//calculate error
	double cur_error = 0;
	for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
	{
		cur_error += user_latent[i] * movie_latent[i];
	}
	cur_error -= edge_rating;

	//take gradient step
	for(unsigned int i = 0; i < LATENT_VECTOR_SIZE; i++)
	{
		double prev_movie_val = movie_latent[i];
		
		double a = step_size * (cur_error * user_latent[i] + LAMBDA * prev_movie_val);
		movie_latent[i] -= a;
		double b = step_size * (cur_error * prev_movie_val + LAMBDA * user_latent[i]);
		user_latent[i] -= b;
	}

	++movie_data.updates;
}

void verify (Graph& g) {
	// computing Root Mean Square Error
	// Assuming only movie nodes have edges

	typedef Galois::GAccumulator<double> AccumDouble;

	AccumDouble rms;

	Galois::do_all_local (g, 
			[&g, &rms] (GNode n) {
			for (auto e = g.edge_begin (n, Galois::NONE)
				, e_end = g.edge_end (n, Galois::NONE); e != e_end; ++e) {

			GNode m = g.getEdgeDst (e);
			double pred = calcPrediction (g.getData (n, Galois::NONE), g.getData (m, Galois::NONE));
			double rating = g.getEdgeData (e, Galois::NONE);
			
			rms += ((pred - rating) * (pred - rating));
			}
			});

	double total_rms = rms.reduce ();
	double final_rms = sqrt(total_rms/NUM_RATINGS);
	
	std::cout << "Root Mean Square Error after training: " << total_rms << " " << final_rms << std::endl;
}

struct sgd_block
{
	Graph& g;
	sgd_block(Graph& g) : g(g) {}

	void operator()(ThreadWorkItem& workItem)
	{
		Galois::Timer timer;
		timer.start();
		int updates = 0;

		unsigned int userRangeEnd = workItem.userRangeEnd;

		//set up movie iterators
		Graph::iterator movie_it = g.begin();
		std::advance(movie_it, workItem.movieRangeStart);
		Graph::iterator end_movie_it = g.begin();
		std::advance(end_movie_it, workItem.movieRangeEnd);

		unsigned prev_updates = updates;
		unsigned last_edge_reached = 0;

		//for each movie in the range
		for(; movie_it != end_movie_it; ++movie_it)
		{
			//get movie data
			GNode movie = *movie_it;
			Node& movie_data = g.getData(movie);

			//for each edge in the range
			Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::NONE) + movie_data.edge_offset;
			Graph::edge_iterator edge_end = g.edge_end(movie, Galois::NONE);
			if(edge_it == edge_end) last_edge_reached++;
			for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
			{
				GNode user = g.getEdgeDst(edge_it);

				//stop when you're outside the current block's user range
				if(user > userRangeEnd)
					break;

				Node& user_data = g.getData(user, Galois::NONE);
				unsigned int edge_rating = g.getEdgeData(edge_it, Galois::NONE);	

				//do gradient step
				doGradientUpdate(movie_data, user_data, edge_rating);

				updates++;
			}

			//we just looked at the last user
			if(userRangeEnd == NUM_USER_NODES)
			{
				//start back at the first edge
				movie_data.edge_offset = 0;
			}
		}

		timer.stop();
		workItem.timeTaken = timer.get_usec();
		workItem.updates = updates;
	}

};

struct sgd_block_users
{
	Graph& g;
	sgd_block_users(Graph& g) : g(g) {}

	void operator()(ThreadWorkItem& workItem)
	{
		Galois::Timer timer;
		timer.start();
		int updates = 0;

		unsigned int usersPerBlockSlice = workItem.usersPerBlockSlice;
		unsigned int currentBlockSliceEnd = workItem.userRangeStart;
		unsigned int userRangeEnd = workItem.userRangeEnd;

		while(currentBlockSliceEnd < userRangeEnd)
		{
			currentBlockSliceEnd += usersPerBlockSlice;
			if(currentBlockSliceEnd > userRangeEnd)
				currentBlockSliceEnd = userRangeEnd;

			//set up movie iterators
			Graph::iterator movie_it = g.begin();
			std::advance(movie_it, workItem.movieRangeStart);
			Graph::iterator end_movie_it = g.begin();
			std::advance(end_movie_it, workItem.movieRangeEnd);

			unsigned prev_updates = updates;
			unsigned last_edge_reached = 0;

			//for each movie in the range
			for(; movie_it != end_movie_it; ++movie_it)
			{
				//get movie data
				GNode movie = *movie_it;
				Node& movie_data = g.getData(movie);

				unsigned int currentBlockSliceEndUserId = currentBlockSliceEnd + NUM_MOVIE_NODES;// + 1;

				//for each edge in the range
				Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::NONE) + movie_data.edge_offset;
				Graph::edge_iterator edge_end = g.edge_end(movie, Galois::NONE);
				if(edge_it == edge_end) last_edge_reached++;
				for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
				{
					GNode user = g.getEdgeDst(edge_it);

					//stop when you're outside the current block's user range
					if(user > currentBlockSliceEndUserId)
						break;

					Node& user_data = g.getData(user, Galois::NONE);
					unsigned int edge_rating = g.getEdgeData(edge_it, Galois::NONE);	

					//do gradient step
					doGradientUpdate(movie_data, user_data, edge_rating);

					updates++;
				}

				//we just looked at the last user
				if(currentBlockSliceEnd == NUM_USER_NODES)
				{
					//start back at the first edge
					movie_data.edge_offset = 0;
				}
			}
		}

		timer.stop();
		workItem.timeTaken = timer.get_usec();
		workItem.updates = updates;
	}

};

//movies blocked also
struct sgd_block_users_movies
{
	Graph& g;
	sgd_block_users_movies(Graph& g) : g(g) {}

	void operator()(ThreadWorkItem& workItem)
	{
		Galois::Timer timer;
		timer.start();
		int updates = 0;

		unsigned int usersPerBlockSlice = workItem.usersPerBlockSlice;
		unsigned int currentBlockSliceEnd = workItem.userRangeStart;
		unsigned int userRangeEnd = workItem.userRangeEnd;

		while(currentBlockSliceEnd < userRangeEnd)
		{
			currentBlockSliceEnd += usersPerBlockSlice;
			if(currentBlockSliceEnd > userRangeEnd)
				currentBlockSliceEnd = userRangeEnd;

			unsigned int currentMovieSliceEnd = workItem.movieRangeStart;
			unsigned int moviesPerBlockSlice = workItem.moviesPerBlockSlice;
			unsigned int movieRangeEnd = workItem.movieRangeEnd;

			while(currentMovieSliceEnd < movieRangeEnd)
			{
				Graph::iterator movie_it = g.begin();
				std::advance(movie_it, currentMovieSliceEnd);

				currentMovieSliceEnd += moviesPerBlockSlice;
				if(currentMovieSliceEnd > movieRangeEnd)
					currentMovieSliceEnd = movieRangeEnd;

				Graph::iterator end_movie_it = g.begin();
				std::advance(end_movie_it, currentMovieSliceEnd);

				//for each movie in the range
				for(; movie_it != end_movie_it; ++movie_it)
				{
					//get movie data
					GNode movie = *movie_it;
					Node& movie_data = g.getData(movie);

					unsigned int currentBlockSliceEndUserId = currentBlockSliceEnd + NUM_MOVIE_NODES;// + 1;

					//for each edge in the range
					Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::NONE) + movie_data.edge_offset;
					Graph::edge_iterator edge_end = g.edge_end(movie, Galois::NONE);
					for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
					{
						GNode user = g.getEdgeDst(edge_it);

						//stop when you're outside the current block's user range
						if(user > currentBlockSliceEndUserId)
							break;

						Node& user_data = g.getData(user, Galois::NONE);
						unsigned int edge_rating = g.getEdgeData(edge_it, Galois::NONE);	

						//do gradient step
						doGradientUpdate(movie_data, user_data, edge_rating);

						updates++;
					}

					//we just looked at the last user
					if(currentBlockSliceEnd == NUM_USER_NODES)
					{
						//start back at the first edge
						movie_data.edge_offset = 0;
					}
				}
			}
		}

		timer.stop();
		workItem.timeTaken = timer.get_usec();
		workItem.updates = updates;
	}

};

unsigned int userIdToUserNode(unsigned int userId)
{
	return userId + NUM_MOVIE_NODES + 1;
}

struct advance_edge_iterators
{
	Graph& g;
	advance_edge_iterators(Graph& g) : g(g) {}

	void operator()(ThreadWorkItem& workItem)
	{
		//set up movie iterators
		Graph::iterator movie_it = g.begin();
		std::advance(movie_it, workItem.movieRangeStart);
		Graph::iterator end_movie_it = g.begin();
		std::advance(end_movie_it, workItem.movieRangeEnd);

		//for each movie in the range
		for(; movie_it != end_movie_it; ++movie_it)
		{
			//get movie data
			GNode movie = *movie_it;
			Node& movie_data = g.getData(movie);

			//for each edge in the range
			Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::NONE) + movie_data.edge_offset;
			Graph::edge_iterator edge_end = g.edge_end(movie, Galois::NONE);
			for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
			{
				GNode user = g.getEdgeDst(edge_it);

				//stop when you're in the acceptable user range
				if(user > userIdToUserNode(workItem.userRangeStart))
					break;
			}
		}
	}
};

//utility function to learn about a graph input
void count_ratings(Graph& g) {
	const unsigned threadCount = Galois::getActiveThreads ();

	std::vector<unsigned long> ratings_per_user(NUM_USER_NODES);
	std::vector<unsigned long> ratings_per_movie(NUM_MOVIE_NODES);

	Graph::iterator movie_it = g.begin();
	Graph::iterator end_movie_it = g.end();

	for(; movie_it != end_movie_it; ++movie_it)
	{
		//get movie data
		GNode movie = *movie_it;
		Node& movie_data = g.getData(movie);

		//for each edge in the range
		Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::NONE) + movie_data.edge_offset;
		Graph::edge_iterator edge_end = g.edge_end(movie, Galois::NONE);
		for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
		{
			GNode user = g.getEdgeDst(edge_it);

			ratings_per_user[user- NUM_MOVIE_NODES]++;
			ratings_per_movie[movie]++;
		}
	}

	//unsigned num_zeroes = 0;
	//for(int i = 0; i < NUM_USER_NODES; i++)
	//  {
	//  if(ratings_per_user[i] == 0) num_zeroes++;
	//std::cout << ratings_per_user[i] << std::endl;
	//}
	//std::cout << "Num zeroes " << num_zeroes << std::endl;

	unsigned int per = NUM_USER_NODES/threadCount;
	for(int i = 0; i < threadCount; i++)
	{
		unsigned start = per * i;
		unsigned end = per * (i+1);
		if(i == threadCount - 1) end = NUM_USER_NODES;

		unsigned count = 0;
		for(int j = start; j < end; j++)
			count += ratings_per_user[j];
		std::cout << i << ": " << count << std::endl;
	}

	/*unsigned num_zeroes = 0;
	  for(int i = 0; i < NUM_MOVIE_NODES; i++)
	  {
	//if(ratings_per_user[i] == 0) num_zeroes++;
	std::cout << ratings_per_movie[i] << std::endl;
	}*/
	//std::cout << "Num zeroes " << num_zeroes << std::endl;
}

template<typename BlockFn>
void runBlockSlices(Graph& g) {

	const unsigned threadCount = Galois::getActiveThreads ();

	unsigned numWorkItems = threadCount;
	ThreadWorkItem workItems[numWorkItems];
	unsigned int moviesPerThread = NUM_MOVIE_NODES / numWorkItems;
	unsigned int usersPerThread = NUM_USER_NODES / numWorkItems;
	unsigned int userRangeStartPoints[numWorkItems];
	unsigned int userRangeEndPoints[numWorkItems];

	//set up initial work ranges for each thread
	for(unsigned int i = 0; i < numWorkItems; i++)
	{
		ThreadWorkItem wi;
		wi.movieRangeStart = moviesPerThread * i;
		wi.userRangeStart = usersPerThread * i;
		//stored for the advance_edge_iterators step
		userRangeStartPoints[i] = wi.userRangeStart;

		if(i == numWorkItems - 1) //last blocks take the rest
		{
			wi.movieRangeEnd = NUM_MOVIE_NODES;
			wi.userRangeEnd = NUM_USER_NODES;
		}
		else
		{
			wi.movieRangeEnd = wi.movieRangeStart + moviesPerThread;
			wi.userRangeEnd = (i+1) * usersPerThread;
		}

		//stored to make it easy to move the blocks assigned to threads
		userRangeEndPoints[i] = wi.userRangeEnd;

		wi.usersPerBlockSlice = usersPerBlockSlice;
		wi.moviesPerBlockSlice = moviesPerBlockSlice;

		//debug vars
		wi.id = i;
		wi.updates = 0;

		workItems[i] = wi;
	}

	//move the edge iterators of each movie to the start of the current block
	//advances the edge iterator until it reaches the userRangeStart field of the ThreadWorkItem
	//userRangeStart isn't needed after this point
	Galois::do_all(workItems + 0, workItems + numWorkItems, advance_edge_iterators(g));

	unsigned long** updates = new unsigned long*[numWorkItems];
	for(int i = 0; i < numWorkItems; i++)
	{
		updates[i] = new unsigned long[numWorkItems];
	}

	//update all movies/users MAX_MOVIE_UPDATES times
	for(unsigned int update = 0; update < MAX_MOVIE_UPDATES; update++)
	{	
		//std::cout << "Iteration " << update << std::endl;
		//verify (g);

		//work on the current blocks, move the block a thread works on to the right
		for(unsigned int j = 0; j < numWorkItems; j++)
		{	
			// std::cout << "Update " << update << " Block " << j << std::endl;
			//assign one ThreadWorkItem to each thread statically
			Galois::do_all(workItems + 0, workItems + numWorkItems, BlockFn(g));
			//Galois::do_all(workItems + 0, workItems + numWorkItems, sgd_block_users_movies(g));

			//move each thread's assignment of work one block to the right
			//(for the same movie nodes, look at the next range of user nodes)
			for(unsigned int k = 0; k < numWorkItems; k++)
			{
				ThreadWorkItem& wi = workItems[k];
				//std::cout << " (" << wi.userRangeStart << "," << wi.userRangeEnd << ") " << (long) (wi.timeTaken/1000) << "/" << (long) wi.updates;
				//std::cout << (long) wi.updates << " ";
				unsigned int column = (j+k)%numWorkItems;
				//updates[k][column] = wi.updates;
				updates[k][column] = wi.timeTaken/1000;
				//std::cout << (long) wi.updates << " ";
				//std::cout << (long) wi.timeTaken << " ";
				unsigned int nextColumn = (j+1+k)%numWorkItems;
				wi.userRangeStart = userRangeStartPoints[nextColumn];
				wi.userRangeEnd = userRangeEndPoints[nextColumn];
			}
			//std::cout << std::endl;
		}
	}

	for(int i = 0; i < numWorkItems; i++)
	{
		for(int j = 0; j < numWorkItems; j++)
			std::cout << updates[i][j] << " ";
		std::cout << std::endl;
	}
}

typedef Galois::Runtime::LL::PaddedLock<true> SpinLock;

struct sgd_march
{
	Graph& g;
	SpinLock* locks; 
	sgd_march(Graph& g, SpinLock* locks) : g(g), locks(locks) {}

	void operator()(ThreadWorkItem& workItem)
	{
		Galois::Timer timer;
		timer.start();
		unsigned int updates = 0;
		unsigned int conflicts = 0;

		unsigned int usersPerBlockSlice = workItem.usersPerBlockSlice;
		unsigned int currentBlockSliceEnd = workItem.userRangeStart;
		unsigned int userRangeEnd = workItem.userRangeEnd;

		unsigned int currentSliceId = workItem.sliceStart;
		unsigned int sliceUpdates = 0;

		while(sliceUpdates < MAX_MOVIE_UPDATES * workItem.numSlices)
		{
			//spinlock_lock(locks[currentSliceId]);

			if(!locks[currentSliceId].try_lock ())
			{
				conflicts++;
				locks[currentSliceId].lock ();
			}

			//if(workItem.id == 0)
			//	printf("Currslice %d Slice from %d to ", currentSliceId, currentBlockSliceEnd);

			currentBlockSliceEnd += usersPerBlockSlice;
			if(currentBlockSliceEnd > userRangeEnd)
				currentBlockSliceEnd = userRangeEnd;

			//if(workItem.id == 0)
			//	printf("%d\n", currentBlockSliceEnd);

			//set up movie iterators
			Graph::iterator movie_it = g.begin();
			std::advance(movie_it, workItem.movieRangeStart);
			Graph::iterator end_movie_it = g.begin();
			std::advance(end_movie_it, workItem.movieRangeEnd);

			//for each movie in the range
			for(; movie_it != end_movie_it; ++movie_it)
			{
				//get movie data
				GNode movie = *movie_it;
				Node& movie_data = g.getData(movie);

				unsigned int currentBlockSliceEndUserId = currentBlockSliceEnd + NUM_MOVIE_NODES;// + 1;

				//printf("movie %d edge_offset %d\n", movie, movie_data.edge_offset);

				//for each edge in the range
				Graph::edge_iterator edge_it = g.edge_begin(movie, Galois::NONE) + movie_data.edge_offset;
				Graph::edge_iterator edge_end = g.edge_end(movie, Galois::NONE);
				for(;edge_it != edge_end; ++edge_it, ++movie_data.edge_offset)
				{
					GNode user = g.getEdgeDst(edge_it);

					//printf("looked at user %d\n", user - NUM_MOVIE_NODES);
					//stop when you're outside the current block's user range
					if(user > currentBlockSliceEndUserId)
						break;
					//printf("okay user %d\n", user - NUM_MOVIE_NODES);

					Node& user_data = g.getData(user, Galois::NONE);
					unsigned int edge_rating = g.getEdgeData(edge_it, Galois::NONE);	

					//do gradient step
					doGradientUpdate(movie_data, user_data, edge_rating);

					updates++;
					//if(workItem.id == 0) printf("yoookay user %d\n", user - NUM_MOVIE_NODES);
					//if(workItem.id == 13) printf("updates %d\n", updates);
				}

				//printf("movie %d edge_offset is now %d\n", movie, movie_data.edge_offset);

				//we just looked at the last user
				if(currentBlockSliceEnd == NUM_USER_NODES)
				{
					//start back at the first edge
					movie_data.edge_offset = 0;
				}
			}

			locks[currentSliceId].unlock ();

			currentSliceId++;
			sliceUpdates++;

			//if(currentSliceId == workItem.numSlices) //hit end of slices
			if(currentBlockSliceEnd == userRangeEnd)
			{
				currentSliceId = 0;
				currentBlockSliceEnd = 0;
			}
		}

		timer.stop();
		workItem.timeTaken = timer.get_usec();
		workItem.updates = updates;
		std::cout << workItem.id << " " << workItem.updates << " " << workItem.timeTaken/1000000 << " " << conflicts << std::endl;
		//printf("Worker %d took %u to do %u updates.\n", workItem.id, workItem.timeTaken/1000000, workItem.updates);
	}

};


void runSliceMarch(Graph& g) {

	const unsigned threadCount = Galois::getActiveThreads ();
	unsigned numWorkItems = threadCount;
	ThreadWorkItem workItems[numWorkItems];
	unsigned int moviesPerThread = NUM_MOVIE_NODES / numWorkItems;
	unsigned int usersPerThread = NUM_USER_NODES / numWorkItems;

	unsigned int numSlices = NUM_USER_NODES / usersPerBlockSlice;

	SpinLock* locks = new SpinLock[numSlices];

	unsigned int slicesPerThread = numSlices / threadCount;
	printf("numSlices: %d slicesPerThread: %d\n", numSlices, slicesPerThread);

	//set up initial work ranges for each thread
	for(unsigned int i = 0; i < numWorkItems; i++)
	{
		ThreadWorkItem wi;
		wi.movieRangeStart = moviesPerThread * i;
		wi.userRangeStart = usersPerThread * i;
		wi.userRangeEnd = NUM_USER_NODES;

		if(i == numWorkItems - 1) //last blocks take the rest
		{
			wi.movieRangeEnd = NUM_MOVIE_NODES;
		}
		else
		{
			wi.movieRangeEnd = wi.movieRangeStart + moviesPerThread;
		}

		wi.sliceStart = slicesPerThread * i;
		wi.numSlices = numSlices;
		wi.usersPerBlockSlice = usersPerBlockSlice;

		//debug vars
		wi.id = i;
		wi.updates = 0;

		workItems[i] = wi;
	}

	//move the edge iterators of each movie to the start of the current block
	//advances the edge iterator until it reaches the userRangeStart field of the ThreadWorkItem
	//userRangeStart isn't needed after this point
	Galois::do_all(workItems + 0, workItems + numWorkItems, advance_edge_iterators(g));
	Galois::do_all(workItems + 0, workItems + numWorkItems, sgd_march(g, locks));


	/*for(unsigned int i = 0; i < numWorkItems; i++)
	  {
	  ThreadWorkItem& workItem = workItems[i];
	  printf("Worker %d took %lu to do %lu updates.\n", workItem.id, workItem.timeTaken/1000000, workItem.updates);
	  }*/
}

static double genRand () {
	// generate a random double in (-1,1)
	return double (2 * std::rand ()) / double (RAND_MAX) - 1;
}

// Initializes latent vector and id for each node
unsigned int initializeGraphData(Graph& g)
{
	// unsigned int seed = 42;
	// std::default_random_engine eng(seed);
	// std::uniform_real_distribution<double> random_lv_value(0, 0.1);
	const unsigned SEED = 4562727;
	std::srand (SEED);

	unsigned int numMovieNodes = 0;
	unsigned int numRatings = 0;

	//for all movie and user nodes in the graph
	for (Graph::iterator i = g.begin(), end = g.end(); i != end; ++i) {
		GNode gnode = *i;
		Node& data = g.getData(gnode);

		data.updates = 0;

		//fill latent vectors with random values
		double* lv = new double[LATENT_VECTOR_SIZE];
		for(int i = 0; i < LATENT_VECTOR_SIZE; i++)
		{
			lv[i] = genRand();
		}
		data.latent_vector = lv;

		//count number of movies we've seen; only movies nodes have edges
		unsigned int num_edges = 
			g.edge_end(gnode, Galois::NONE) - g.edge_begin(gnode, Galois::NONE);
		numRatings += num_edges;
		if(num_edges > 0)
			numMovieNodes++;

		data.edge_offset = 0;
	}

	NUM_RATINGS = numRatings;

	return numMovieNodes;
}



int main(int argc, char** argv) {	

	LonestarStart (argc, argv, name, desc, url);
	Galois::StatManager statManager;

	//allocate local computation graph
	Graph g;

	//read structure of graph & edge weights; nodes not initialized
	Galois::Graph::readGraph(g, inputFile);

	//fill each node's id & initialize the latent vectors
	unsigned int numMovieNodes = initializeGraphData(g);
	unsigned int numUserNodes = g.size() - numMovieNodes;

	std::cout << "Input initialized, num users = " << numUserNodes 
		<< ", num movies = " << numMovieNodes << std::endl;

	NUM_MOVIE_NODES = numMovieNodes;
	NUM_USER_NODES = numUserNodes;

	Galois::StatTimer timer;
	timer.start();

	switch (algo) {

		case Algo::block:
			runBlockSlices<sgd_block>(g);
			break;

		case Algo::blockAndSliceUsers:
			runBlockSlices<sgd_block_users>(g);
			break;

		case Algo::blockAndSliceBoth:
			runBlockSlices<sgd_block_users_movies>(g);
			break;

		case Algo::sliceMarch:
			runSliceMarch(g);
			break;

	}

	timer.stop();

	verify (g);

	std::cout << "SUMMARY Movies " << numMovieNodes << 
		" Users " << numUserNodes <<
		" Ratings " << g.sizeEdges() << 
		" usersPerBlockSlice " << usersPerBlockSlice << 
		" moviesPerBlockSlice " << moviesPerBlockSlice << 
		" Time " << timer.get()/1000.0 << std::endl;

	return 0;
}
