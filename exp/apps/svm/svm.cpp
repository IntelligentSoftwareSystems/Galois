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
 * Stochastic gradient descent for solving linear SVM, implemented with Galois.
 *
 * Author: Prad Nelluru <pradn@cs.utexas.edu>
*/

#include <iostream>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <vector>

#include "Galois/config.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/Graph.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Accumulator.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Lonestar/BoilerPlate.h"

/**                       CONFIG                       **/

static const char* const name = "Stochastic Gradient Descent for Linear Support Vector Machines";
static const char* const desc = "Implements a linear support vector machine using stochastic gradient descent";
static const char* const url = "sgdsvm";

namespace cll = llvm::cl;
static cll::opt<std::string> inputGraphFilename(cll::Positional, cll::desc("<graph input file>"), cll::Required);
static cll::opt<std::string> inputLabelFilename(cll::Positional, cll::desc("<label input file>"), cll::Required);
static cll::opt<double> CREG("c", cll::desc("the regularization parameter C"), cll::init(1.0));
static cll::opt<bool> NO_LOCKS("nl", cll::desc("do not lock feature nodes"), cll::init(false));
static cll::opt<bool> BOUNDED_STALENESS("bs", cll::desc("use bounded staleness looking back 1 iteration"), cll::init(false));
static cll::opt<bool> SHUFFLE("s", cll::desc("shuffle samples between iterations"), cll::init(false));
static cll::opt<double> TRAINING_FRACTION("tr", cll::desc("fraction of samples to use for training"), cll::init(0.8));
static cll::opt<double> ACCURACY_GOAL("ag", cll::desc("accuracy at which to stop running"), cll::init(0.95));
static cll::opt<unsigned> ITER("i", cll::desc("how many iterations to run for, ignoring accuracy the goal"), cll::init(0));

/**                      DATA TYPES                    **/

typedef struct Node
{
	double w; //weight - relevant for variable nodes
	int field; //variable nodes - variable count, sample nodes - label
} Node;

using Graph = Galois::Graph::LC_CSR_Graph<Node, double>;
using GNode = Graph::GraphNode;

/**               CONSTANTS AND PARAMETERS             **/
unsigned NUM_SAMPLES = 0;
unsigned NUM_VARIABLES = 0;

unsigned variableNodeToId(GNode variable_node)
{
	return ((unsigned) variable_node) - NUM_SAMPLES;
}

double w_at(Graph& g, unsigned windex)
{
	return g.getData(windex + NUM_SAMPLES).w; 
}

double* old_weights;

template <bool USE_BOUNDED_STALENESS = false>
struct linearSVM
{	
	Graph& g;
	double learningRate;
	Galois::MethodFlag lock_mode;

	linearSVM(Graph& _g, double _lr) : g(_g), learningRate(_lr) 
	{
		lock_mode = NO_LOCKS ? Galois::NONE : Galois::ALL;
	}
	
	void operator()(GNode gnode, Galois::UserContext<GNode>& ctx)
	{	
		double dot = 0.0;
		Galois::MethodFlag mode = lock_mode;
		for(auto edge_it : g.out_edges(gnode, mode))
		{
			GNode variable_node = g.getEdgeDst(edge_it);
			
			Node& var_data = g.getData(variable_node, mode);

			double weight;
			if(!USE_BOUNDED_STALENESS)
				weight = var_data.w; //normal algorithm
			else 
				weight = old_weights[variableNodeToId(variable_node)]; //bounded staleness

			dot += weight * g.getEdgeData(edge_it, mode);
		}
		
		Node& sample_data = g.getData(gnode, mode);
		int label = sample_data.field;

		bool update_type = label * dot < 1;
		for(auto edge_it : g.out_edges(gnode, mode))
		{
			GNode variable_node = g.getEdgeDst(edge_it);
			Node& var_data = g.getData(variable_node, mode);
			int varCount = var_data.field;
			
			double weight;
			if(!USE_BOUNDED_STALENESS)
				weight = var_data.w; //normal algorithm
			else
				weight = old_weights[variableNodeToId(variable_node)]; //bounded staleness
			
			double delta;
			if(update_type)
				delta = learningRate * ( weight/( CREG * varCount ) - label * g.getEdgeData(edge_it, mode));
			else
				delta = weight/( CREG * varCount);

			//printf("%f\n", delta);
			var_data.w -= delta;
		}
	}
};

void printParameters()
{
	std::cout << "Input graph file: " << inputGraphFilename << std::endl;
	std::cout << "Input label file: " << inputLabelFilename << std::endl;
	std::cout << "Threads: " << Galois::getActiveThreads() << std::endl;
	std::cout << "Samples: " << NUM_SAMPLES << std::endl;
	std::cout << "Variables: " << NUM_VARIABLES << std::endl;
}

void initializeGraphData(Graph& g)
{
	for (auto gnode : g) 
	{
		Node& data = g.getData(gnode);
		data.w = 0.0;
		data.field = 0;
	}
}

void initializeVariableCounts(Graph& g)
{
	for (auto gnode : g) {
		Galois::MethodFlag mode = Galois::NONE;
		for(auto edge_it : g.out_edges(gnode))
		{
			GNode variable_node = g.getEdgeDst(edge_it);
			Node& data = g.getData(variable_node);
			data.field++; //increase count of variable occurrences
		}
	}
}

void loadLabels(Graph& g, std::string filename)
{
	std::ifstream infile(filename);

	unsigned sample_id;
	int label;
	int num_labels = 0;
	while(infile >> sample_id >> label)
	{
		g.getData(sample_id).field = label;
		++num_labels;
	}
	
	NUM_SAMPLES = num_labels;
}

double getAccuracy(Graph& g, std::vector<GNode>& testing_samples)
{
	Galois::MethodFlag mode = Galois::NONE;
	unsigned correct = 0;
	for (auto gnode : testing_samples) {
		double sum = 0.0;
		Node& data = g.getData(gnode);
		int label = data.field;
		
		if(gnode <= NUM_SAMPLES)
		{
			for(auto edge_it : g.out_edges(gnode, mode))
			{
				GNode variable_node = g.getEdgeDst(edge_it);
				Node& data = g.getData(variable_node, mode);
				double weight = g.getEdgeData(edge_it, mode);
				sum += data.w * weight;
			}

			//std::cout << "Gnode: " << gnode << " Sum: " << sum << " Label: " << label;
			if(sum <= 0.0 && label == -1)
			{
				//std::cout << " CORRECT";
				correct++;
			}
			else if(sum > 0.0 && label == 1)
			{
				//std::cout << " CORRECT";
				correct++;
			}
			//else std::cout << " INCORRECT";

			//std::cout << std::endl;
		}
	}
	
	double accuracy = correct / (testing_samples.size() + 0.0);
	std::cout << "Accuracy: " << accuracy << " (" << correct <<  "/" << testing_samples.size() << ")" << std::endl;
	return accuracy;
}

//find how many documents conflict with a particular document
void analyze(Graph& g)
{
	Galois::MethodFlag mode = Galois::NONE;
	std::vector<GNode> badnode2s{ NUM_SAMPLES, 0 };
	//std::cout << "Size: " << badnode2s.size() << std::endl;

	for(GNode node1 : g)
	{
		if(node1 == NUM_SAMPLES)
			break;

		//std::cout << node1 << ": " << std::endl;

		unsigned count = 0;
		for(GNode node2 : g)
		{
			if(node1 == node2)
				continue;
			else if(node2 == NUM_SAMPLES)
				break;

			auto n1b = g.edge_begin(node1, mode);
			auto n1e = g.edge_end(node1, mode);
			
			auto n2b = g.edge_begin(node2, mode);
			auto n2e = g.edge_end(node2, mode);
			
			bool conflict = false;
			while(n1b != n1e && n2b != n2e)
			{
				GNode v1 = g.getEdgeDst(n1b);
				GNode v2 = g.getEdgeDst(n2b);

				if(v1 == v2)
				{
					//std::cout << "Conflict between " << node1 << " and " << node2 << std::endl;
					++count;
					conflict = true;
					++badnode2s[node2];
					break;
				}
				else if(v1 < v2)
					++n1b;
				else
					++n2b;
			}

			//if(!conflict)
			//	std::cout << " " << node2;
		}
		
		//std::cout << std::endl;

		//std::cout << "Sample " << node1 << ": " << count << 
		//	" (" <<  ((count+0.0)/NUM_SAMPLES) << ")" << std::endl;
	}

	for(auto count : badnode2s)
		std::cout << count << std::endl;

	exit(-1);
}

int main(int argc, char** argv)
{
	LonestarStart(argc, argv, name, desc, url);
	Galois::StatManager statManager;
	
	Graph g;
	std::cout << "done with loading 0" << std::endl;
	Galois::Graph::readGraph(g, inputGraphFilename);
	std::cout << "done with loading 1" << std::endl;
	
	initializeGraphData(g);
	loadLabels(g, inputLabelFilename); //also sets NUM_SAMPLES
	initializeVariableCounts(g);
	
	//analyze(g);

	NUM_VARIABLES = g.size() - NUM_SAMPLES;
	assert(NUM_SAMPLES > 0 && NUM_VARIABLES > 0);
	
	//put samples in a list and shuffle them
	std::vector<GNode> all_samples(g.begin(), g.begin() + NUM_SAMPLES);
	std::random_shuffle(all_samples.begin(), all_samples.end());

	//copy a fraction of the samples to the training samples list
	unsigned num_training_samples = NUM_SAMPLES * TRAINING_FRACTION;
	std::vector<GNode> training_samples(all_samples.begin(), all_samples.begin() + num_training_samples);
	std::cout << "Training samples: " << training_samples.size() << std::endl;
	
	//the remainder of samples go into the testing samples list
	std::vector<GNode> testing_samples(all_samples.begin() + num_training_samples, all_samples.end());
	std::cout << "Testing samples: " << testing_samples.size() << std::endl;
	
	//allocate storage for weights from previous iteration
	if(BOUNDED_STALENESS)
	{
		old_weights = new double[NUM_VARIABLES]; //global
		std::fill(old_weights, old_weights + NUM_VARIABLES, 0);
	}

	printParameters();
	getAccuracy(g, testing_samples);
	
	Galois::TimeAccumulator timer;
	
	//if no iteration count is specified, keep going until the accuracy goal is hit
	//	otherwise, run specified iterations
	const bool use_accuracy_goal = ITER == 0;
	double accuracy = -1.0; //holds most recent accuracy stat
	for(unsigned iter = 1; iter <= ITER || use_accuracy_goal; iter++)
	{
		timer.start();
		
		//include shuffling time in the time taken per iteration
		//also: not parallel
		if(SHUFFLE)
			std::random_shuffle(training_samples.begin(), training_samples.end());
		
		double learning_rate = 30/(100.0 + iter);
		auto ts_begin = training_samples.begin();
		auto ts_end = training_samples.end();
		auto loopname = Galois::loopname("Linear SVM");
		auto worklist = Galois::wl<Galois::WorkList::dChunkedFIFO<32>>();
		if(!BOUNDED_STALENESS)
			Galois::for_each(ts_begin, ts_end, linearSVM<false>(g, learning_rate), loopname, worklist);
		else
			Galois::for_each(ts_begin, ts_end, linearSVM<true>(g, learning_rate), loopname, worklist);

		timer.stop();

		std::cout << iter << " ";
		
		accuracy = getAccuracy(g, testing_samples);
		if(use_accuracy_goal && accuracy > ACCURACY_GOAL)
		{
			std::cout << "Accuracy goal of " << ACCURACY_GOAL << " reached after " <<
				iter << " iterations."<< std::endl;
			break;
		}
		
		//swap weights from past iteration and this iteration
		if(BOUNDED_STALENESS)
			for(unsigned i = 0; i < NUM_SAMPLES; i++)
			{
				GNode variable_node = (GNode) (i + NUM_SAMPLES);
				Node& var_data = g.getData(variable_node, Galois::NONE);
				double weight = var_data.w;
				old_weights[variableNodeToId(variable_node)] = weight;
			}
	}
	
	double timeTaken = timer.get()/1000.0;
	std::cout << "Time: " << timeTaken << std::endl;

	return 0;
}
