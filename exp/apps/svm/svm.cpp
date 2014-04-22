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

enum class UpdateType {
        Wild,
        ReplicateByThread,
        ReplicateByPackage,
        Staleness
};

namespace cll = llvm::cl;
static cll::opt<std::string> inputGraphFilename(cll::Positional, cll::desc("<graph input file>"), cll::Required);
static cll::opt<std::string> inputLabelFilename(cll::Positional, cll::desc("<label input file>"), cll::Required);
static cll::opt<double> CREG("c", cll::desc("the regularization parameter C"), cll::init(1.0));
static cll::opt<bool> NO_LOCKS("nl", cll::desc("do not lock feature nodes"), cll::init(false));
static cll::opt<bool> SHUFFLE("s", cll::desc("shuffle samples between iterations"), cll::init(false));
static cll::opt<double> TRAINING_FRACTION("tr", cll::desc("fraction of samples to use for training"), cll::init(0.8));
static cll::opt<double> ACCURACY_GOAL("ag", cll::desc("accuracy at which to stop running"), cll::init(0.95));
static cll::opt<unsigned> ITER("i", cll::desc("how many iterations to run for, ignoring accuracy the goal"), cll::init(0));
static cll::opt<UpdateType> UPDATE_TYPE("algo", cll::desc("Update type:"),
        cll::values(
                clEnumValN(UpdateType::Wild, "wild", "unsynchronized (default)"),
                clEnumValN(UpdateType::ReplicateByThread, "replicateByThread", "thread replication"),
                clEnumValN(UpdateType::ReplicateByPackage, "replicateByPackage", "package replication"),
                clEnumValN(UpdateType::Staleness, "staleness", "stale reads"),
                clEnumValEnd), cll::init(UpdateType::Wild));


/**                      DATA TYPES                    **/

typedef struct Node
{
	double w; //weight - relevant for variable nodes
	int field; //variable nodes - variable count, sample nodes - label
        Node(): w(0.0), field(0) { }
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

Galois::Runtime::PerThreadStorage<double*> thread_weights;
Galois::Runtime::PerPackageStorage<double*> package_weights;
Galois::LargeArray<double> old_weights;

template <UpdateType UT>
struct linearSVM
{	
	Graph& g;
	double learningRate;
	Galois::MethodFlag lock_mode;
        bool has_other;
	linearSVM(Graph& _g, double _lr) : g(_g), learningRate(_lr) 
	{
		lock_mode = NO_LOCKS ? Galois::NONE : Galois::ALL;
                unsigned threads_per_package = 8;
                has_other = Galois::getActiveThreads() > threads_per_package;
	}
	
	void operator()(GNode gnode, Galois::UserContext<GNode>& ctx)
	{	
		double dot = 0.0;
                double *packagew = *package_weights.getLocal();
                double *threadw = *thread_weights.getLocal();
                double *otherw = NULL;
                unsigned tid = Galois::Runtime::LL::getTID();
                if (has_other) 
                {
                        unsigned my_package = Galois::Runtime::LL::getPackageForSelf(tid);
                        unsigned next = my_package + 1;
                        if (next >= 4)
                                next -= 4;
                        otherw = *package_weights.getRemoteByPkg(next);
                }
		Galois::MethodFlag mode = lock_mode;
		for(auto edge_it : g.out_edges(gnode, mode))
		{
			GNode variable_node = g.getEdgeDst(edge_it);
			
			Node& var_data = g.getData(variable_node, mode);

			double weight;
                        switch (UT) {
                                case UpdateType::Wild:
                                        weight = var_data.w; //normal algorithm
                                        break;
                                case UpdateType::ReplicateByThread:
                                        weight = threadw[variableNodeToId(variable_node)];
                                        break;
                                case UpdateType::ReplicateByPackage:
                                        weight = packagew[variableNodeToId(variable_node)];
                                        break;
                                case UpdateType::Staleness:
                                        weight = old_weights[variableNodeToId(variable_node)]; //bounded staleness
                                        break;
                                default: abort();
                        }

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
                        switch (UT) {
                                case UpdateType::Wild:
                                        weight = var_data.w; //normal algorithm
                                        break;
                                case UpdateType::ReplicateByThread:
                                        weight = threadw[variableNodeToId(variable_node)];
                                        break;
                                case UpdateType::ReplicateByPackage:
                                        weight = packagew[variableNodeToId(variable_node)];
                                        break;
                                case UpdateType::Staleness:
                                        weight = old_weights[variableNodeToId(variable_node)]; //bounded staleness
                                        break;
                                default: abort();
                        }
			
			double delta;
			if(update_type)
				delta = learningRate * ( weight/( CREG * varCount ) - label * g.getEdgeData(edge_it, mode));
			else
				delta = weight/( CREG * varCount);

                        switch (UT) {
                                case UpdateType::Wild:
                                        var_data.w -= delta;
                                        break;
                                case UpdateType::ReplicateByThread:
                                        threadw[variableNodeToId(variable_node)] -= delta;
                                        break;
                                case UpdateType::ReplicateByPackage:
                                        packagew[variableNodeToId(variable_node)] -= delta;
                                        if (otherw && update_type)
                                                otherw[variableNodeToId(variable_node)] -= delta;
                                        break;
                                case UpdateType::Staleness:
                                        threadw[variableNodeToId(variable_node)] -= delta;
                                        break;
                                default: abort();
                        }
		}
	}
};

void printParameters()
{
	std::cout << "Input graph file: " << inputGraphFilename << "\n";
	std::cout << "Input label file: " << inputLabelFilename << "\n";
	std::cout << "Threads: " << Galois::getActiveThreads() << "\n";
	std::cout << "Samples: " << NUM_SAMPLES << "\n";
	std::cout << "Variables: " << NUM_VARIABLES << "\n";
        switch (UPDATE_TYPE) {
                case UpdateType::Wild:
                        std::cout << "Update type: wild\n";
                        break;
                case UpdateType::ReplicateByThread:
                        std::cout << "Update type: replicate by thread\n";
                        break;
                case UpdateType::ReplicateByPackage:
                        std::cout << "Update type: replicate by package\n";
                        break;
                case UpdateType::Staleness:
                        std::cout << "Update type: stale reads\n";
                        break;
                default: abort();
        }
}

void initializeVariableCounts(Graph& g)
{
        for (auto gnode : g) 
        {
                for(auto edge_it : g.out_edges(gnode))
                {
                        GNode variable_node = g.getEdgeDst(edge_it);
                        Node& data = g.getData(variable_node);
                        data.field++; //increase count of variable occurrences
                }
        }
}


unsigned loadLabels(Graph& g, std::string filename)
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
	
	return num_labels;
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

			if(sum <= 0.0 && label == -1)
			{
				correct++;
			}
			else if(sum > 0.0 && label == 1)
			{
				correct++;
			}
		}
	}
	
	double accuracy = correct / (testing_samples.size() + 0.0);
	std::cout << "Accuracy: " << accuracy << " (" << correct <<  "/" << testing_samples.size() << ")" << std::endl;
	return accuracy;
}

int main(int argc, char** argv)
{
	LonestarStart(argc, argv, name, desc, url);
	Galois::StatManager statManager;
	
	Graph g;
	Galois::Graph::readGraph(g, inputGraphFilename);
	NUM_SAMPLES = loadLabels(g, inputLabelFilename);
	initializeVariableCounts(g);
	
	NUM_VARIABLES = g.size() - NUM_SAMPLES;
	assert(NUM_SAMPLES > 0 && NUM_VARIABLES > 0);
	
	//put samples in a list and shuffle them
	std::vector<GNode> all_samples(g.begin(), g.begin() + NUM_SAMPLES);
	std::random_shuffle(all_samples.begin(), all_samples.end());

	//copy a fraction of the samples to the training samples list
	unsigned num_training_samples = NUM_SAMPLES * TRAINING_FRACTION;
	std::vector<GNode> training_samples(all_samples.begin(), all_samples.begin() + num_training_samples);
	std::cout << "Training samples: " << training_samples.size() << "\n";
	
	//the remainder of samples go into the testing samples list
	std::vector<GNode> testing_samples(all_samples.begin() + num_training_samples, all_samples.end());
	std::cout << "Testing samples: " << testing_samples.size() << "\n";
	
	//allocate storage for weights from previous iteration
        old_weights.create(NUM_VARIABLES);
        if(UPDATE_TYPE == UpdateType::ReplicateByThread || UPDATE_TYPE == UpdateType::Staleness) 
	{
                Galois::on_each([](unsigned tid, unsigned total) {
                        double *p = new double[NUM_VARIABLES];
                        *thread_weights.getLocal() = p;
                        std::fill(p, p + NUM_VARIABLES, 0);
                });
	}
        if(UPDATE_TYPE == UpdateType::ReplicateByPackage)
        {
                Galois::on_each([](unsigned tid, unsigned total) {
                        if (Galois::Runtime::LL::isPackageLeader(tid)) {
                                double *p = new double[NUM_VARIABLES];
                                *package_weights.getLocal() = p;
                                std::fill(p, p + NUM_VARIABLES, 0);
                        }
                });
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
		auto ln = Galois::loopname("LinearSVM");
		auto wl = Galois::wl<Galois::WorkList::dChunkedFIFO<32>>();

                UpdateType type = UPDATE_TYPE;
                switch (type) {
                        case UpdateType::Wild:
                                Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::Wild>(g, learning_rate), ln, wl);
                                break;
                        case UpdateType::ReplicateByPackage:
                                Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::ReplicateByPackage>(g, learning_rate), ln, wl);
                                break;
                        case UpdateType::ReplicateByThread:
                                Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::ReplicateByThread>(g, learning_rate), ln, wl);
                                break;
                        case UpdateType::Staleness:
                                Galois::for_each(ts_begin, ts_end, linearSVM<UpdateType::Staleness>(g, learning_rate), ln, wl);
                                break;
                        default: abort();
                }

		timer.stop();

		//swap weights from past iteration and this iteration
		if(type != UpdateType::Wild) 
                {
                        bool byThread = type == UpdateType::ReplicateByThread || type == UpdateType::Staleness;
                        double *localw = byThread ? *thread_weights.getLocal() : *package_weights.getLocal();
			for(unsigned i = 0; i < NUM_VARIABLES; i++)
			{
                                unsigned num_threads = Galois::getActiveThreads();
                                unsigned threads_per_package = 8;
                                unsigned num_packages = (Galois::getActiveThreads() + threads_per_package - 1) / threads_per_package;
                                unsigned n = byThread ? num_threads : num_packages;
                                for (unsigned j = 1; j < n; j++)
                                {
                                        double o = byThread ?
                                                (*thread_weights.getRemote(j))[i] :
                                                (*package_weights.getRemoteByPkg(j))[i];
                                        localw[i] += o; //(o - old_weights[i]);
                                }
				GNode variable_node = (GNode) (i + NUM_SAMPLES);
				Node& var_data = g.getData(variable_node, Galois::NONE);
                                var_data.w = localw[i];
				old_weights[i] = var_data.w;
			}
                        Galois::on_each([&](unsigned tid, unsigned total) {
                                switch (type) {
                                        case UpdateType::Staleness:
                                        case UpdateType::ReplicateByThread:
                                                if (tid)
                                                        std::copy(localw, localw + NUM_VARIABLES, *thread_weights.getLocal());
                                        break;
                                        case UpdateType::ReplicateByPackage:
                                                if (tid && Galois::Runtime::LL::isPackageLeader(tid))
                                                        std::copy(localw, localw + NUM_VARIABLES, *package_weights.getLocal());
                                        break;
                                }
                        });
                }

		std::cout << iter << " ";
		accuracy = getAccuracy(g, testing_samples);
		if(use_accuracy_goal && accuracy > ACCURACY_GOAL)
		{
			std::cout << "Accuracy goal of " << ACCURACY_GOAL << " reached after " <<
				iter << " iterations."<< std::endl;
			break;
		}
	}
	
	double timeTaken = timer.get()/1000.0;
	std::cout << "Time: " << timeTaken << std::endl;

	return 0;
}
// vim: ts=8 sw=8
