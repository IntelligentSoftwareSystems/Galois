#ifndef MULTILABEL_H
#define MULTILABEL_H

#include "bilinear.h"

enum {ML_LS, ML_LR, ML_L2SVC};

class multilabel_problem {
	public:
	bilinear_problem *training_set;
	bilinear_problem *test_set;
	multilabel_problem(bilinear_problem *training_set, bilinear_problem *test_set) {
		this->training_set = training_set;
		this->test_set = test_set;
	}
};

class multilabel_parameter: public bilinear_parameter{ 
	public:
		int maxiter;
		int top_p;
		int k;
		int threads;
		int reweighting;
		bool predict;
		// Parameters for Wsabie
		double lrate; // learning rate for wsabie
		multilabel_parameter() {
			bilinear_parameter();
			reweighting = 0;
			maxiter = 10; 
			top_p = 20; 
			k = 10;
			threads = 8;
			lrate = 0.01;
			predict = true;
		}
};

#ifdef __cplusplus
extern "C" {
#endif

void multilabel_train(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H);

#ifdef __cplusplus
}
#endif


#endif
