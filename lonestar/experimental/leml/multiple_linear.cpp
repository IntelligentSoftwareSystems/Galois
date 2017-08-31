
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include "multiple_linear.h"
#include "tron.h"
#include "dmat.h"
#ifdef EXP_DOALL_GALOIS
#include "Galois/Galois.h"
#endif

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void print_null(const char *s) {}
static void (*liblinear_print_string) (const char *) = &print_string_stdout;



/*
struct dense_problem {
	smat_t *Y;
	double *X;
	int target_col;
	int k;
};
*/


class l2r_dense_lr_fun: public function { 
protected:
	const double *y_val;
	const unsigned *y_idx;
	const double *X;
	size_t w_size;
	size_t l;

	double *C;
	double *z;
	double *D;
	double *wa;
public:
	l2r_dense_lr_fun(const smat_t &Y, const double *X, int w_size, int target_col, double *C) { 
		y_val = &Y.val[Y.col_ptr[target_col]];
		y_idx = &Y.row_idx[Y.col_ptr[target_col]];
		this->X = X;
		this->w_size = w_size;
		l = Y.nnz_of_col(target_col);
		this->C = C;
		z = new double[l];
		D = new double[l];
		wa = new double[l];
	}
	~l2r_dense_lr_fun(){
		delete[] z;
		delete[] D;
		delete[] wa;
	}

	void Xv(double *v, double *Xv){
		for(size_t i = 0; i < l; i++) {
			const double *Xi = &X[y_idx[i]*w_size];
			Xv[i] = 0;
			for(unsigned t = 0; t < w_size; t++)
				Xv[i] += Xi[t]*v[t];
		}
	}

	void XTv(double *v, double *XTv){
		for(unsigned t = 0; t < w_size; t++) 
			XTv[t] = 0;
		for(size_t i = 0; i < l; i++) {
			const double *Xi = &X[y_idx[i]*w_size];
			for(unsigned t = 0; t < w_size; t++)
				XTv[t] += Xi[t]*v[i];
		}
	}

	double fun(double *w) {
		double f=0;
		Xv(w, z);
		for(size_t i=0;i<w_size;i++)
			f += w[i]*w[i];
		f /= 2.0;
		for(size_t i=0;i<l;i++) {
			double yz = y_val[i]*z[i];
			if (yz >= 0)
				f += C[i]*log(1 + exp(-yz));
			else
				f += C[i]*(-yz+log(1 + exp(yz)));
		}
		return(f);
	}

	void grad(double *w, double *g) {
		for(size_t i=0;i<l;i++) {
			z[i] = 1/(1 + exp(-y_val[i]*z[i]));
			D[i] = z[i]*(1-z[i]);
			z[i] = C[i]*(z[i]-1)*y_val[i];
		}
		XTv(z, g);
		for(size_t i=0;i<w_size;i++)
			g[i] = w[i] + g[i];
	}
	void Hv(double *s, double *Hs) {
		Xv(s, wa);
		for(size_t i=0;i<l;i++)
			wa[i] = C[i]*D[i]*wa[i];
		XTv(wa, Hs);
		for(size_t i=0;i<w_size;i++)
			Hs[i] = s[i] + Hs[i];
	}
	int get_nr_variable(void) {return (int)w_size;}
}; 

class l2r_dense_l2svc_fun: public function { 
protected:
	const double *y_val;
	const unsigned *y_idx;
	const double *X;
	size_t w_size;
	size_t l;

	double *C;
	double *z;
	double *D;
	double *wa;
	int *I;
	size_t sizeI;
public:
	l2r_dense_l2svc_fun(const smat_t &Y, const double *X, int w_size, int target_col, double *C) { 
		y_val = &Y.val[Y.col_ptr[target_col]];
		y_idx = &Y.row_idx[Y.col_ptr[target_col]];
		this->X = X;
		this->w_size = w_size;
		l = Y.nnz_of_col(target_col);
		this->C = C;
		z = new double[l];
		D = new double[l];
		wa = new double[l];
		I = new int[l];
	}
	~l2r_dense_l2svc_fun(){
		delete[] z;
		delete[] D;
		delete[] wa;
		delete[] I;
	}

	void Xv(double *v, double *Xv){
		for(size_t i = 0; i < l; i++) {
			const double *Xi = &X[y_idx[i]*w_size];
			Xv[i] = 0;
			for(unsigned t = 0; t < w_size; t++)
				Xv[i] += Xi[t]*v[t];
		}
	}

	void XTv(double *v, double *XTv){
		for(unsigned t = 0; t < w_size; t++) 
			XTv[t] = 0;
		for(size_t i = 0; i < l; i++) {
			const double *Xi = &X[y_idx[i]*w_size];
			for(unsigned t = 0; t < w_size; t++)
				XTv[t] += Xi[t]*v[i];
		}
	}

	void subXv(double *v, double *Xv) {
		for(size_t i = 0; i < sizeI; i++) {
			const double *Xi = &X[y_idx[I[i]]*w_size];
			Xv[i] = 0;
			for(unsigned t = 0; t < w_size; t++)
				Xv[i] += Xi[t]*v[t];
		}
	}

	void subXTv(double *v, double *XTv){
		for(unsigned t = 0; t < w_size; t++) 
			XTv[t] = 0;
		for(size_t i = 0; i < sizeI; i++) {
			const double *Xi = &X[y_idx[I[i]]*w_size];
			for(unsigned t = 0; t < w_size; t++)
				XTv[t] += Xi[t]*v[i];
		}
	}

	double fun(double *w) {
		double f=0;
		Xv(w, z);
		for(size_t i=0;i<w_size;i++)
			f += w[i]*w[i];
		f /= 2.0;
		for(size_t i=0;i<l;i++) {
			z[i] = y_val[i]*z[i];
			double d = 1-z[i];
			if(d > 0) {
				f += C[i]*d*d;
			}
		}
		return(f);
	}

	void grad(double *w, double *g) {
		sizeI = 0;
		for(size_t i=0;i<l;i++) {
			if(z[i] < 1) {
				z[sizeI] = C[i]*y_val[i]*(z[i]-1);
				I[sizeI++] = (int)i;
			}
		}
		subXTv(z, g);
		for(size_t i=0;i<w_size;i++)
			g[i] = w[i] + 2*g[i];
	}

	void Hv(double *s, double *Hs) {
		subXv(s, wa);
		for(size_t i=0;i<sizeI;i++)
			wa[i] = C[I[i]]*wa[i];

		subXTv(wa, Hs);
		for(size_t i=0;i<w_size;i++)
			Hs[i] = s[i] + 2*Hs[i];
	}
	int get_nr_variable(void) {return (int)w_size;}
}; 

class l2r_dense_ls_fun: public function{
protected:
	const double *y_val;
	const unsigned *y_idx;
	const double *X;
	size_t w_size;
	size_t l;

	double *C;
	double *z;
	double *D;
public:
	l2r_dense_ls_fun(const smat_t &Y, const double *X, int w_size, int target_col, double *C) { 
		this->X = X;
		this->w_size = w_size;
		y_val = &Y.val[Y.col_ptr[target_col]];
		y_idx = &Y.row_idx[Y.col_ptr[target_col]];
		l = Y.nnz_of_col(target_col);
		z = new double[l];
		D = new double[l];
		this->C = C;
	}
	~l2r_dense_ls_fun(){
		delete[] z;
		delete[] D;
	}
	void Xv(double *v, double *Xv){
		for(size_t i = 0; i < l; i++) {
			const double *Xi = &X[y_idx[i]*w_size];
			Xv[i] = 0;
			for(unsigned t = 0; t < w_size; t++)
				Xv[i] += Xi[t]*v[t];
		}
	}
	void XTv(double *v, double *XTv){
		for(unsigned t = 0; t < w_size; t++) 
			XTv[t] = 0;
		for(size_t i = 0; i < l; i++) {
			const double *Xi = &X[y_idx[i]*w_size];
			for(unsigned t = 0; t < w_size; t++)
				XTv[t] += Xi[t]*v[i];
		}
	}
	double fun(double *w) {
		double f=0;
		Xv(w, z);
		for(size_t i=0;i<w_size;i++)
			f += w[i]*w[i];
		f /= 2.0;

		for(size_t i=0;i<l;++i){
			double tmp=(y_val[i]-z[i]);
			f += C[i]*tmp*tmp;
		}
		return(f);
	}
	void grad(double *w, double *g) {
		for(size_t i=0;i<l;++i)
			z[i] = C[i]*(z[i]-y_val[i]);
		XTv(z, g);
		for(size_t i=0;i<w_size;++i)
			g[i] = 2.0*g[i] + w[i];
	}
	void Hv(double *s, double *Hs) {
		Xv(s, D);
		for(size_t i=0;i<l;++i) 
			D[i]*=C[i];
		XTv(D, Hs);
		for(size_t i=0;i<w_size;i++)
			Hs[i] = 2.0*Hs[i] + s[i];
	}
	int get_nr_variable(void) {return (int)w_size;}
};


int multiple_l2r_ls_chol_full_weight(multiple_linear_problem *prob, multiple_linear_parameter *param, double *W){
	smat_t &Y = *(prob->Y), Yt; Yt = Y.transpose();
	double *X = prob->X;
	const int k = (int)prob->k;
	const double lambda = 1.0; 
	double Cp = param->Cp, Cn = param->Cn;
	double alpha = Cp-Cn, beta = Cn;
	double *fixed_Hessian = MALLOC(double, k*k);
	double time = omp_get_wtime();

	// W = 2*(alpha+beta) * Y^T X
	smat_x_dmat(Yt, X, k, W);
	do_axpy(2*(alpha+beta)-1.0, W, W, Y.cols * k);

	// fixed_Hessian = 2*beta * X^T X + lambda * I
	doHTH(X, fixed_Hessian, Y.rows, k);
	do_axpy(2*beta-1.0, fixed_Hessian, fixed_Hessian, k*k);

	for(int i = 0; i < k; i++)
		fixed_Hessian[i*k+i] += lambda;

	int nr_threads = omp_get_max_threads();
	double** Hessian_set = MALLOC(double*, nr_threads);
	for(int i = 0; i < nr_threads; i++)
		Hessian_set[i] = MALLOC(double, k*k);

#ifdef EXP_DOALL_GALOIS
	Galois::do_all(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(Y.cols),
            [&](size_t j) {
#else
#pragma omp parallel for schedule(dynamic,50) shared(Y,W,X)
	for(size_t j = 0; j < Y.cols; ++j) {
#endif
		long nnz_j = Y.col_ptr[j+1] - Y.col_ptr[j];
		if(nnz_j) {
#ifdef EXP_DOALL_GALOIS
                  int tid = Galois::Substrate::ThreadPool::getTID();
#else
                        int tid = omp_get_thread_num(); // thread ID
#endif
                        double *Wj = &W[j*k];
                        double *Hessian = Hessian_set[tid];
                        memcpy(Hessian, fixed_Hessian, sizeof(double)*k*k);

                        // Update Hessian 
                        for(long idx = Y.col_ptr[j]; idx < Y.col_ptr[j+1]; ++idx){
                                const double *Xi = &X[k*Y.row_idx[idx]];
                                for(int s = 0; s < k; ++s) {
                                        for(int t = s; t < k; ++t)
                                                Hessian[s*k+t]+=2*alpha*Xi[s]*Xi[t];
                                }
                        }
                        for(int s = 0; s < k; ++s)
                                for(int t = 0; t < s; ++t)
                                        Hessian[s*k+t] = Hessian[t*k+s];
                        ls_solve_chol_matrix(Hessian, Wj, k);
                }
#ifdef EXP_DOALL_GALOIS
        });
#else
	}
#endif

	printf("half-als: %lg secs\n", omp_get_wtime() - time);

	for(int i = 0; i < nr_threads; i++)
		free(Hessian_set[i]);
	free(Hessian_set);
	free(fixed_Hessian);
	return 0;
}


/*
 *  W = argmin_{W}  C * |Y - X*W'|^2 +  0.5*|W|^2
 *    = argmin_{W} 0.5*|Y - X*W'|^2 + 0.5*lambda*|W|^2
 *    lambda = 1/2C
 */
int multiple_l2r_ls_chol_full(multiple_linear_problem *prob, multiple_linear_parameter *param, double *W){
	smat_t &Y = *(prob->Y), Yt; Yt = Y.transpose();
	double *X = prob->X;
	const int k = (int)prob->k;
	double lambda = 1.0/(2.0*param->Cp);
	double *XTX = MALLOC(double, k*k);
	double time = omp_get_wtime();
	smat_x_dmat(Yt, X, k, W);
	doHTH(X, XTX, Y.rows, k);
	for(int i = 0; i < k; i++)
		XTX[i*k+i] += lambda;
	ls_solve_chol_matrix(XTX, W, k, Y.cols);
	printf("half-als: %lg secs\n", omp_get_wtime() - time);
	free(XTX);
	return 0;
}

int multiple_l2r_ls_chol(multiple_linear_problem *prob, multiple_linear_parameter *param, double *W){
	smat_t &Y = *(prob->Y);
	const double *X = prob->X;
	const int k = prob->k;
	size_t nnz = Y.nnz;
	const double lambda = 1.0; //(param->Cp);
	double *C = MALLOC(double, nnz);
	double time = omp_get_wtime();

	for(size_t idx = 0; idx < nnz; idx++) {
		if (Y.val[idx] > 0) {
			C[idx] = 2*param->Cp; 
		} else {
			C[idx] = 2*param->Cn;
		}
	}

	int nr_threads = omp_get_max_threads();
	double** Hessian_set = MALLOC(double*, nr_threads);
	for(int i = 0; i < nr_threads; i++)
		Hessian_set[i] = MALLOC(double, k*k);

#ifdef EXP_DOALL_GALOIS
	Galois::do_all(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(Y.cols),
            [&](size_t j) {
#else
#pragma omp parallel for schedule(dynamic,50) shared(Y,W,X)
	for(size_t j = 0; j < Y.cols; ++j) {
#endif
		long nnz_j = Y.col_ptr[j+1] - Y.col_ptr[j];
		if(nnz_j) {
#ifdef EXP_DOALL_GALOIS
                        int tid = Galois::Substrate::ThreadPool::getTID();
#else
                        int tid = omp_get_thread_num(); // thread ID
#endif
                        double *Wj = &W[j*k];
                        double *Hessian = Hessian_set[tid]; 
                        double *y = Wj; //MALLOC(double,k);
                        memset(Hessian, 0, sizeof(double)*k*k);
                        memset(y, 0, sizeof(double)*k);

                        // Construct k*k Hessian and k*1 y
                        for(long idx = Y.col_ptr[j]; idx < Y.col_ptr[j+1]; ++idx){
                                const double *Xi = &X[k*Y.row_idx[idx]];
                                for(int s = 0; s < k; ++s) {
                                        y[s] += C[idx]*Y.val[idx]*Xi[s];
                                        for(int t = s; t < k; ++t)
                                                Hessian[s*k+t] += C[idx]*Xi[s]*Xi[t];
                                }
                        }
                        for(int s = 0; s < k; ++s) {
                                for(int t = 0; t < s; ++t)
                                        Hessian[s*k+t] = Hessian[t*k+s];
                                Hessian[s*k+s] += lambda;
                        }
                        ls_solve_chol_matrix(Hessian, y, k);
                }
#ifdef EXP_DOALL_GALOIS
        });
#else
	}
#endif
	printf("half-als: %lg secs\n", omp_get_wtime() - time);
	for(int i = 0; i < nr_threads; i++)
		free(Hessian_set[i]);
	free(Hessian_set);
	return 0;
}

int multiple_l2r_ls_tron(multiple_linear_problem *prob, multiple_linear_parameter *param, double *W) {
	smat_t &Y = *(prob->Y);
	const double *X = prob->X;
	const int k = prob->k;
	size_t nnz = Y.nnz, pos = 0, neg = 0;
	double *C = MALLOC(double, nnz);
	double time = omp_get_wtime();
	for(size_t i = 0; i < nnz; i++) {
		if (Y.val[i] > 0) {
			C[i] = param->Cp; 
			pos++;
		} else {
			C[i] = param->Cn;
			neg++;
		}
	}

	if(!param->verbose) 
		liblinear_print_string = print_null;

#ifdef EXP_DOALL_GALOIS
	Galois::do_all(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(Y.cols),
            [&](unsigned j) {
#else
#pragma omp parallel for schedule(dynamic,50) shared(Y,W,X)
	for(unsigned j = 0; j < Y.cols; ++j) {
#endif
		long nnz_j = Y.col_ptr[j+1] - Y.col_ptr[j];
		if(nnz_j) {
                        double *Wj = &W[j*k];
                        double *Cj = &C[Y.col_ptr[j]];
                        double primal_solver_classification_tol = param->eps*(double)std::max(std::min(pos,neg),1UL)/(double)nnz_j;
                        primal_solver_classification_tol = param->eps;
                        l2r_dense_ls_fun fun_obj = l2r_dense_ls_fun(Y, X, k, j, Cj);
                        TRON tron_obj(&fun_obj, primal_solver_classification_tol, param->max_tron_iter, param->max_cg_iter); 
                        tron_obj.set_print_string(liblinear_print_string);
                        tron_obj.tron(Wj, false); 
                }
#ifdef EXP_DOALL_GALOIS
        });
#else
	}
#endif
	printf("half-least-squre: %lg secs\n", omp_get_wtime() - time);

	return 0;
}

int multiple_l2r_lr_tron(multiple_linear_problem *prob, multiple_linear_parameter *param, double *W) {
	smat_t &Y = *(prob->Y);
	const double *X = prob->X;
	const int k = prob->k;
	size_t nnz = Y.nnz, pos = 0, neg = 0;
	double *C = MALLOC(double, nnz);
	double time = omp_get_wtime();
	for(size_t i = 0; i < nnz; i++) {
		if (Y.val[i] > 0) {
			C[i] = param->Cp; 
			pos++;
		} else {
			C[i] = param->Cn;
			neg++;
		}
	}
	if(!param->verbose) 
		liblinear_print_string = print_null;

#ifdef EXP_DOALL_GALOIS
	Galois::do_all(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(Y.cols),
            [&](unsigned j) {
#else
#pragma omp parallel for schedule(dynamic,50) shared(Y,W,X)
	for(unsigned j = 0; j < Y.cols; ++j) {
#endif
		long nnz_j = Y.col_ptr[j+1] - Y.col_ptr[j];
		if(nnz_j) {
                        double *Wj = &W[j*k];
                        double *Cj = &C[Y.col_ptr[j]];
                        double primal_solver_classification_tol = param->eps*(double)std::max(std::min(pos,neg),1UL)/(double)nnz_j;
                        primal_solver_classification_tol = param->eps;
                        l2r_dense_lr_fun fun_obj = l2r_dense_lr_fun(Y, X, k, j, Cj);
                        TRON tron_obj(&fun_obj, primal_solver_classification_tol, param->max_tron_iter, param->max_cg_iter); 
                        tron_obj.set_print_string(liblinear_print_string);
                        tron_obj.tron(Wj, false); 
                }
#ifdef EXP_DOALL_GALOIS
        });
#else
	}
#endif
	printf("half-logistic-regression: %lg secs\n", omp_get_wtime() - time);

	return 0;
}


int multiple_l2r_l2svc_tron(multiple_linear_problem *prob, multiple_linear_parameter *param, double *W) {
	smat_t &Y = *(prob->Y);
	const double *X = prob->X;
	const int k = prob->k;
	size_t nnz = Y.nnz, pos = 0, neg = 0;
	double *C = MALLOC(double, nnz);
	double time = omp_get_wtime();
	for(size_t i = 0; i < nnz; i++) {
		if (Y.val[i] > 0) {
			C[i] = param->Cp; 
			pos++;
		} else {
			C[i] = param->Cn;
			neg++;
		}
	}
	if(!param->verbose) 
		liblinear_print_string = print_null;

#ifdef EXP_DOALL_GALOIS
	Galois::do_all(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(Y.cols),
            [&](unsigned j) {
#else
#pragma omp parallel for schedule(dynamic,50) shared(Y,W,X)
	for(unsigned j = 0; j < Y.cols; ++j) {
#endif
		long nnz_j = Y.col_ptr[j+1] - Y.col_ptr[j];
		if(nnz_j) {
                        double *Wj = &W[j*k];
                        double *Cj = &C[Y.col_ptr[j]];
                        double primal_solver_classification_tol = param->eps*(double)std::max(std::min(pos,neg),1UL)/(double)nnz_j;
                        primal_solver_classification_tol = param->eps;
                        l2r_dense_l2svc_fun fun_obj = l2r_dense_l2svc_fun(Y, X, k, j, Cj);
                        TRON tron_obj(&fun_obj, primal_solver_classification_tol, param->max_tron_iter, param->max_cg_iter); 
                        tron_obj.set_print_string(liblinear_print_string);
                        tron_obj.tron(Wj, false); 
                }
#ifdef EXP_DOALL_GALOIS
        });
#else
	}
#endif
	printf("half-svm: %lg secs\n", omp_get_wtime() - time);

	return 0;
}
