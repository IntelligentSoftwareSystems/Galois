#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unistd.h>

#include "multilabel.h"
#include "bilinear.h"

#ifdef EXP_DOALL_GALOIS
#include "Galois/Galois.h"
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
void exit_with_help()
{
	printf(
	"Usage: train-ml [options] data-dir/\n"
	"options:\n"
	"    -s type : set type of solver (default 0)\n"    
	"    	 0 -- L2R_LS (Squared Loss)\n"    
	"    	 1 -- L2R_LR (Logistic Regression)\n"    
	"    	 2 -- L2R_SVC (Squared Hinge Loss)\n"    
	"    	 10 -- L2R_LS (Squared Loss) Fully observation\n"    
	"    	 11 -- L2R_LS (Squared Loss) Fully observation with reweighting\n"    
	"    	 20 -- L2R_WSABIE (WAR Loss) length+sample\n"    
	"    	 21 -- L2R_WSABIE_2 (WAR Loss) norm+permuation\n"    
	"    	 22 -- L2R_WSABIE_3 (WAR Loss) length+sample+full_rank\n"    
	"    	 23 -- L2R_WSABIE_4 (WAR Loss) norm+permuation+full_rank\n"    
	"    	 24 -- L2R_WSABIE (WAR Loss) rank estimation\n"    
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 8)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
	"    -t max_iter: set the number of iterations (default 10)\n"    
	"    -T max_tron_iter: set the number of iterations used in TRON (default 5)\n"    
	"    -g max_cg_iter: set the number of iterations used in CG (default 20)\n"
	"    -e epsilon : set inner termination criterion epsilon of TRON (default 0.1)\n"     
	"    -r learning_rate: set the learning rate for WSABIE (default 0.1)\n"
	"    -w reweighting: apply reweight (default 0)\n"
	"    -P top-p: set top-p accruacy (default 20)\n"
	"    -q show_predict: set top-p accruacy (default 1)\n"
	);
	exit(1);
}

multilabel_parameter parse_command_line(int argc, char **argv, char *Y_src, char *X_src, char *Yt_src, char *Xt_src)
{
	multilabel_parameter param;   // default values have been set by the constructor 
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.Cp = 1/(atof(argv[i]));
				param.Cn = param.Cp;
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.max_tron_iter = atoi(argv[i]);
				break;

			case 'g':
				param.max_cg_iter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'r':
				param.lrate = atof(argv[i]);
				break;

			case 'w':
				param.reweighting = atoi(argv[i]);
				break;

			case 'P':
				param.top_p = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	// determine filenames
	if(i>=argc)
		exit_with_help();

	sprintf(Y_src, "%s/Y.smat", argv[i]);
	sprintf(X_src, "%s/X.smat", argv[i]);
	sprintf(Yt_src, "%s/Yt.smat", argv[i]);
	sprintf(Xt_src, "%s/Xt.smat", argv[i]);

	if(param.solver_type == L2R_BILINEAR_LS_FULL)
		param.reweighting = 0;

	return param;
}

/*
void rand_init_old(double *M, int m, int n, std::default_random_engine &gen) {
	std::normal_distribution<double> distribution (0.0,1.0);
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			M[i*n+j] = distribution(gen);
}
*/

double* rand_init(double *M, int m, int n) {
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			M[i*n+j] = drand48();
	return M;
}
void run_multilabel_train(multilabel_parameter param, smat_t &Y, smat_t &X, smat_t &Yt, smat_t &Xt) {
	srand48(0UL);
//	std::default_random_engine generator(0);
	int k = param.k;
	double *W=NULL, *H=NULL; 
	if(param.reweighting)
		param.reweighting_based_on(Y);
	printf("Cp => %g, Cn => %g\n", param.Cp, param.Cn);

	if(param.solver_type == L2R_WSABIE3 || param.solver_type == L2R_WSABIE4) {
		k = param.k = (int)X.cols;
		printf("k = %d\n", k);
		H = Malloc(double, k*Y.cols);
		//rand_init(H, (int)Y.cols, k, generator);
		rand_init(H, (int)Y.cols, k);
	} else {
		W = Malloc(double, k*X.cols);
		H = Malloc(double, k*Y.cols);
		//rand_init(W, (int)X.cols, k, generator);
		//rand_init(H, (int)Y.cols, k, generator);
		rand_init(W, (int)X.cols, k);
		rand_init(H, (int)Y.cols, k);
	}

	bilinear_problem training_set(&Y, &X, H, k);
	bilinear_problem test_set(&Yt, &Xt, H, k);
	multilabel_problem prob(&training_set, &test_set);
	omp_set_num_threads(param.threads);

        //hack to force omp threads to be created before galois stuff
#pragma omp parallel for
	for(size_t i = 0; i < 1000; i++) {
          sleep(0);
        }

#ifdef EXP_DOALL_GALOIS
        galois::setActiveThreads(param.threads);
        galois::substrate::getThreadPool().burnPower(param.threads);
#endif
	std::vector<int> subset = subsample((int)(0.01*(double)Y.rows), (int)Y.rows);
	smat_t subX = X.row_subset(subset);
	smat_t subY = Y.row_subset(subset);
	bilinear_problem sub_training_set(&subY, &subX, H, k);
	multilabel_parameter sub_param = param; 
	sub_param.maxiter = 3;
	sub_param.top_p = 5;
	sub_param.predict = false;

	multilabel_problem subprob(&sub_training_set, &test_set);
	multilabel_train(&subprob, &sub_param, W, H);
	multilabel_train(&prob, &param, W, H);

	if(W) free(W);
	if(H) free(H);
}

int main(int argc, char* argv[]){
	char X_src[1024], Xt_src[1024], Y_src[1024], Yt_src[1024];
	char hostname[1024];
	if(gethostname(hostname, 1024)!=0) 
		puts("Cannot get the hostname!");
	else 
		printf("Running on Host: %s\n", hostname);
	for(int i = 0; i < argc; i++)
		printf("%s ", argv[i]);
	puts("");
	smat_t Y,Yt,X,Xt;
	multilabel_parameter param = parse_command_line(argc, argv, Y_src, X_src, Yt_src, Xt_src); 
	X.load_from_binary(X_src);
	Y.load_from_binary(Y_src); 
	Yt.load_from_binary(Yt_src); 
	Xt.load_from_binary(Xt_src);

	run_multilabel_train(param, Y, X, Yt, Xt); 
	return 0;
}

