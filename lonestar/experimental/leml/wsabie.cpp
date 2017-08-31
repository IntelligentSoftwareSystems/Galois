
#include "wsabie.h"

#include <map>
#ifdef EXP_DOALL_GALOIS
#include "Galois/Galois.h"
#endif

using namespace std;

static double norm(double *W, size_t size) {
	double ret = 0;
	for(size_t i = 0; i < size; i++)
		ret += W[i]*W[i];
	return sqrt(ret);
}
// WSABIE: Scaling Up To Large Vocabulary Image Annotation, 2011
// Jason Weston and Samy Bengio and Nicolas Usunier 

// W and H are stored in row-majored order
// return x_i^T W h_j, and Wxi
static double cal_score(smat_t& X, double *W, double *H, size_t k, long i, long j, double *Wxi = NULL) {
	if(Wxi!=NULL) {
		memset(Wxi, 0, sizeof(double)*k);
		for(long idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++)
			do_axpy(X.val_t[idx], W+X.col_idx[idx]*k, Wxi, k);
		return do_dot_product(H+j*k, Wxi, k);
	} else {
		double ret = 0.0;
		for(long idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++) {
			ret += X.val_t[idx] * do_dot_product(W+X.col_idx[idx]*k, H+j*k, k);
		}
		return ret;
	}
}

static inline void project_with_len(double *w, long k, double wsabieC) {
	double rescale = 0;
	for(int t = 0; t < k; t++) rescale += w[t]*w[t];
	if(rescale > wsabieC*wsabieC) {
		rescale = wsabieC / sqrt(rescale);
		for(int t = 0; t < k; t++) w[t] *= rescale;
	}
}

double get_wsabieC(const multilabel_parameter *param) {
	// Cp is the variance, 3 makes 99.7% tolerance intervals
	return 3*sqrt(param->Cp);
}

void wsabie_model_projection(double *W, double *H, long nr_feats, long nr_labels, long k, double wsabieC) {
#ifdef EXP_DOALL_GALOIS
	Galois::do_all(boost::counting_iterator<int>(0), boost::counting_iterator<int>(nr_feats),
            [&](int s) {
#else
#pragma omp parallel for schedule(static,50) shared(W)
	for(int s = 0; s < nr_feats; s++) {
#endif
		project_with_len(W+s*k, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
        });
#else
        }
#endif
#ifdef EXP_DOALL_GALOIS
	Galois::do_all(boost::counting_iterator<int>(0), boost::counting_iterator<int>(nr_labels),
            [&](int j) {
#else
#pragma omp parallel for schedule(static,50) shared(H)
	for(int j = 0; j < nr_labels; j++) {
#endif
		project_with_len(H+j*k, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
        });
#else
        }
#endif
}

void wsabie_updates(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples){
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld\n", nr_samples);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	double *Wxi = MALLOC(double, k);
	double *hdiff = MALLOC(double, k);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	// Projection of W and H
	double wsabieC = get_wsabieC(param);
	//wsabie_model_projection(W, H, nr_feats, nr_labels, k, wsabieC);


	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = rand() % (int)nr_insts;
		if (Y.nnz_of_row(int(i)) == 0)  continue;
		int y = Y.col_idx[Y.row_ptr[i] + rand() % Y.nnz_of_row((int)i)], ybar;
		double score_y = cal_score(X, W, H, k, i, y, Wxi), score_ybar = 0.0, L_tri = -1;

		// Estimate Approximate Rank: N
		int N = 0;
		while (1) {
			N++;
			ybar = rand() % (int)(nr_labels-1);
			if(ybar == y) ybar = (int)nr_labels - 1;
			score_ybar = do_dot_product(Wxi, H+ybar*k, k);
			L_tri = 1 - score_y + score_ybar;
			if(L_tri > 0 or (N >= (nr_labels-1))) break;
		}

		if(L_tri > 0) {
			double eta = param->lrate;
			double steplength = L[(nr_labels-1)/N]*eta;
			double *Hy = H+y*k, *Hybar = H+ybar*k;
			// Update W
			for(int t = 0; t < k; t++) 
				hdiff[t] = steplength*(Hybar[t]-Hy[t]);
#ifdef EXP_DOALL_GALOIS
                      Galois::do_all(boost::counting_iterator<long>(X.row_ptr[i]), boost::counting_iterator<long>(X.row_ptr[i+1]),
                          [&](long idx) {
#else
#pragma omp parallel for schedule(static,50) shared(X,hdiff)
			for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
#endif
				double *Ws = W+X.col_idx[idx]*k;
				for(int t = 0; t < k; t++) {
					Ws[t] -= X.val_t[idx]*hdiff[t];
				}
				project_with_len(Ws, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
                        });
#else
			}
#endif

			// Update H_ybar
			do_axpy(-steplength, Wxi, Hybar, k);
			// Update H_y
			do_axpy(steplength, Wxi, Hy, k);
			// Projection
			double len;
			len = sqrt(do_dot_product(Hybar, Hybar, k));
			if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hybar, Hybar, k);
			len = sqrt(do_dot_product(Hy, Hy, k));
			if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hy, Hy, k);
		}
	}
	free(hdiff);
	free(Wxi);
	free(L);
}

void wsabie_updates_new_old(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples){
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld\n", nr_samples);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	double *Wxi = MALLOC(double, k);
	double *hdiff = MALLOC(double, k);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	// Projection of W and H
	double wsabieC = get_wsabieC(param);
	//wsabie_model_projection(W, H, nr_feats, nr_labels, k, wsabieC);

	map<long,bool>pos_labels;

	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = rand() % (int)nr_insts;
		if (Y.nnz_of_row(int(i)) == 0)  continue;
		int y = Y.col_idx[Y.row_ptr[i] + rand() % Y.nnz_of_row((int)i)], ybar;
		double score_y = cal_score(X, W, H, k, i, y, Wxi), score_ybar = 0.0, L_tri = -1;

		pos_labels.clear();
		for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++)
			pos_labels[Y.col_idx[idx]] = true;

		// Estimate Approximate Rank: N
		int N = 0;
		while (true) {
			N++;
			int nn = 0;
			while(true) {
				nn ++;
				ybar = rand() % (int)nr_labels;
				if(pos_labels.find(ybar) == pos_labels.end())
					break;
				if(nn >= nr_labels) {
					N = (int)nr_labels;
					break;
				}
			}
			score_ybar = do_dot_product(Wxi, H+ybar*k, k);
			L_tri = 1 - score_y + score_ybar;
			if(L_tri > 0 or (N >= (nr_labels-1))) break;
		}

		if(L_tri > 0) {
			double eta = param->lrate;
			double steplength = L[(nr_labels-Y.nnz_of_row((int)i))/N]*eta;
			double *Hy = H+y*k, *Hybar = H+ybar*k;
			// Update W
			for(int t = 0; t < k; t++) 
				hdiff[t] = steplength*(Hybar[t]-Hy[t]);
#ifdef EXP_DOALL_GALOIS
                      Galois::do_all(boost::counting_iterator<long>(X.row_ptr[i]), boost::counting_iterator<long>(X.row_ptr[i+1]),
                          [&](long idx) {
#else
#pragma omp parallel for schedule(static,50) shared(X,hdiff)
			for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
#endif
				double *Ws = W+X.col_idx[idx]*k;
				for(int t = 0; t < k; t++) {
					Ws[t] -= X.val_t[idx]*hdiff[t];
				}
				project_with_len(Ws, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
                        });
#else
			}
#endif

			// Update H_ybar
			do_axpy(-steplength, Wxi, Hybar, k);
			// Update H_y
			do_axpy(steplength, Wxi, Hy, k);
			// Projection
			double len;
			len = sqrt(do_dot_product(Hybar, Hybar, k));
			if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hybar, Hybar, k);
			len = sqrt(do_dot_product(Hy, Hy, k));
			if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hy, Hy, k);
		}
	}
	free(hdiff);
	free(Wxi);
	free(L);
}

// New wsabie
class ScoreComp{
	public:
		const double *score;
		ScoreComp(const double *score_) : score(score_){}
		bool operator() (int x, int y) const { return score[x] < score[y]; }
};

void wsabie_updates_new(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples) {
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld  k = %ld\n", nr_samples, k);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	double *Wxi = MALLOC(double, k);
	double *hdiff = MALLOC(double, k);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	// Projection of W and H
	double wsabieC = get_wsabieC(param);
	//wsabie_model_projection(W, H, nr_feats, nr_labels, k, wsabieC);

	vector<int> y_neg(nr_labels);
	vector<int> y_pos(nr_labels);
	vector<int> sorted_y_idx(nr_labels);
	vector<int> bucket_cnts(nr_labels+1);
	vector<double> sorted_y_score(nr_labels);

	for(long j = 0; j < nr_labels; j++)
		y_neg[j] = (int)j;

	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = rand() % (int)nr_insts; 
		// selected instance
		long nr_pos = Y.nnz_of_row(int(i));
		long nr_neg = nr_labels - nr_pos;
		if (nr_pos == 0)  continue;

		for(int iter = 0; iter < 1; iter++) {
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				long j = idx - Y.row_ptr[i];
				int y = Y.col_idx[idx];
				y_pos[j] = y;
			}
			std::stable_sort(y_pos.begin(), y_pos.begin() + nr_pos);
			// Construct y_neg such that all negative indices are in the front of y_neg 
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			int y = Y.col_idx[Y.row_ptr[i] + rand() % Y.nnz_of_row((int)i)], ybar;
			double score_y = cal_score(X, W, H, k, i, y, Wxi), score_ybar = 0.0, L_tri = -1;
			int N = 0;
			while (true) {
				N++;
				ybar = y_neg[rand() % nr_neg];
				score_ybar = do_dot_product(Wxi, H+ybar*k, k);
				L_tri = 1 - score_y + score_ybar;
				if(L_tri > 0 or (N >= (nr_neg))) break;
			}

			// Recover y_neg such that y_neg == 0,...,nr_labels-1
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			if(L_tri > 0) {
				double eta = param->lrate/(double)nr_pos;
				double steplength = L[(nr_labels-Y.nnz_of_row((int)i))/N]*eta;
				double *Hy = H+y*k, *Hybar = H+ybar*k;
				// Update W
				for(int t = 0; t < k; t++) 
					hdiff[t] = steplength*(Hybar[t]-Hy[t]);
#ifdef EXP_DOALL_GALOIS
                                Galois::do_all(boost::counting_iterator<long>(X.row_ptr[i]), boost::counting_iterator<long>(X.row_ptr[i+1]),
                                    [&](long idx) {
#else
#pragma omp parallel for schedule(static,50) shared(X,hdiff)
				for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
#endif
					double *Ws = W+X.col_idx[idx]*k;
					for(int t = 0; t < k; t++) {
						Ws[t] -= X.val_t[idx]*hdiff[t];
					}
					project_with_len(Ws, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
                                });
#else
				}
#endif

				// Update H_ybar
				do_axpy(-steplength, Wxi, Hybar, k);
				// Update H_y
				do_axpy(steplength, Wxi, Hy, k);
				// Projection
				double len;
				len = sqrt(do_dot_product(Hybar, Hybar, k));
				if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hybar, Hybar, k);
				len = sqrt(do_dot_product(Hy, Hy, k));
				if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hy, Hy, k);
			}
		}
	}
	free(hdiff);
	free(Wxi);
	free(L);
}

void wsabie_updates_new2(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples) {
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld  k = %ld\n", nr_samples, k);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	double *Wxi = MALLOC(double, k);
	double *hdiff = MALLOC(double, k);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	// Projection of W and H
	double wsabieC = get_wsabieC(param);
	//wsabie_model_projection(W, H, nr_feats, nr_labels, k, wsabieC);

	vector<int> y_neg(nr_labels);
	vector<int> y_pos(nr_labels);
	vector<int> sorted_y_idx(nr_labels);
	vector<int> bucket_cnts(nr_labels+1);
	vector<double> sorted_y_score(nr_labels);

	for(long j = 0; j < nr_labels; j++)
		y_neg[j] = (int)j;

	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = rand() % (int)nr_insts; 
		// selected instance
		long nr_pos = Y.nnz_of_row(int(i));
		long nr_neg = nr_labels - nr_pos;
		if (nr_pos == 0)  continue;

		for(int iter = 0; iter < 1; iter++) {
			bucket_cnts[nr_pos] = 0;
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				long j = idx - Y.row_ptr[i];
				int y = Y.col_idx[idx];
				y_pos[j] = y;
				bucket_cnts[j] = 0;
				sorted_y_idx[j] = (int)j;
				sorted_y_score[j] = (j==0)? cal_score(X,W,H,k,i,y,Wxi): do_dot_product(Wxi,H+y*k,k);
			}
			// construct sorted_y_score and sorted_y_idx
			std::stable_sort(sorted_y_idx.begin(), sorted_y_idx.begin() + nr_pos, ScoreComp(&sorted_y_score[0]));
			std::stable_sort(sorted_y_score.begin(), sorted_y_score.begin() + nr_pos);
			for(long j = 0; j < nr_pos; j++)
				sorted_y_idx[j] = Y.col_idx[Y.row_ptr[i]+sorted_y_idx[j]];

			std::stable_sort(y_pos.begin(), y_pos.begin() + nr_pos);
			// Construct y_neg such that all negative indices are in the front of y_neg 
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			// Sampling negative labels for N trials
			//int N = 3*nr_pos;
			//int N = nr_labels - 1;
			int N = (int)std::min((double)nr_labels-1.0, (double)nr_pos*(3+log2(nr_pos)))+1;
			int ybar = -1;
			int bucket_id = -1;
			double ybar_score = -10e6; //(*std::min_element(sorted_y_score.begin(), sorted_y_score.begin()+nr_pos))-2.0;

			for(long j = 0; j < N; j++) {
				int ytmp = y_neg[rand() % nr_neg];
				double score_ytmp = do_dot_product(Wxi, H+ytmp*k, k);
				int bid = (int)(std::lower_bound(sorted_y_score.begin(), sorted_y_score.begin()+nr_pos, score_ytmp+2.0) - sorted_y_score.begin());
				bucket_cnts[bid] += 1;
				if(score_ytmp > ybar_score) { // select ybar with largest score
					ybar = ytmp;
					ybar_score = score_ytmp;
					bucket_id = bid;
				}
			}

			// Recover y_neg such that y_neg == 0,...,nr_labels-1
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			if(bucket_cnts[0] == N or ybar == -1) continue;

			double eta = param->lrate/(double)nr_pos;

			memset(hdiff, 0, sizeof(double)*k);
			for(long j = nr_pos; j > 0; j--)
				bucket_cnts[j-1] += bucket_cnts[j];

			// Stochastic Gradient Update 
			double sum_Lj = 0;
			for(long j = bucket_id-1; j >= 0; j--){
				double Lj = L[(int)(nr_neg*bucket_cnts[j+1]/N)];
				int y = sorted_y_idx[j];
				double *Hy = H+y*k;
				// Update hdiff
				do_axpy(-Lj, Hy, hdiff, k);
				//Update H_y
				do_axpy(eta*Lj, Wxi, Hy, k);
				project_with_len(Hy, k, wsabieC);
				sum_Lj += Lj;
			}

			// Update hdiff
			double *Hybar = H + ybar*k;
			do_axpy(sum_Lj, Hybar, hdiff, k);

			// Update H_ybar
			do_axpy(-eta*sum_Lj, Wxi, Hybar, k);
			project_with_len(Hybar, k, wsabieC);

#ifdef EXP_DOALL_GALOIS
                        Galois::do_all(boost::counting_iterator<long>(X.row_ptr[i]), boost::counting_iterator<long>(X.row_ptr[i+1]),
                            [&](long idx) {
#else
#pragma omp parallel for schedule(static,50) shared(X,hdiff)
			for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
#endif
				double *Ws = W+X.col_idx[idx]*k;
				for(int t = 0; t < k; t++) {
					Ws[t] -= eta*X.val_t[idx]*hdiff[t];
				}
				project_with_len(Ws, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
                        });
#else
			}
#endif
		}
	}
	free(hdiff);
	free(Wxi);
	free(L);
}

void wsabie_updates_new4(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples){
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld  k = %ld\n", nr_samples, k);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	double *Wxi = MALLOC(double, k);
	double *hdiff = MALLOC(double, k);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	// Projection of W and H
	double wsabieC = get_wsabieC(param);
	//wsabie_model_projection(W, H, nr_feats, nr_labels, k, wsabieC);

	vector<int> y_neg(nr_labels);
	vector<int> y_pos(nr_labels);
	vector<int> sorted_y_idx(nr_labels);
	vector<int> bucket_cnts(nr_labels+1);
	vector<int> bucket_sample(nr_labels+1);
	vector<double> sorted_y_score(nr_labels);

	for(long j = 0; j < nr_labels; j++)
		y_neg[j] = (int)j;

	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = rand() % (int)nr_insts; // selected instance
		long nr_pos = Y.nnz_of_row(int(i));
		long nr_neg = nr_labels - nr_pos;
		if (nr_pos == 0)  continue;

		for(int iter = 0; iter < 1; iter++) {
			bucket_cnts[nr_pos] = 0;
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				long j = idx - Y.row_ptr[i];
				int y = Y.col_idx[idx];
				y_pos[j] = y;
				bucket_cnts[j] = 0;
				bucket_sample[j] = -1;
				sorted_y_idx[j] =(int) j;
				sorted_y_score[j] = (j==0)? cal_score(X,W,H,k,i,y,Wxi): do_dot_product(Wxi,H+y*k,k);
			}
			// construct sorted_y_score and sorted_y_idx
			std::stable_sort(sorted_y_idx.begin(), sorted_y_idx.begin() + nr_pos, ScoreComp(&sorted_y_score[0]));
			std::stable_sort(sorted_y_score.begin(), sorted_y_score.begin() + nr_pos);
			for(long j = 0; j < nr_pos; j++)
				sorted_y_idx[j] = Y.col_idx[Y.row_ptr[i]+sorted_y_idx[j]];

			std::stable_sort(y_pos.begin(), y_pos.begin() + nr_pos);
			// Construct y_neg such that all negative indices are in the front of y_neg 
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			int y_fake = y_pos[rand() % nr_pos];

			// Sampling negative labels for N trials
			//int N = 3*nr_pos;
			//int N = nr_labels - 1;
			int N = (int)std::min((double)nr_labels-1.0, (double)nr_pos*(3+log2(nr_pos)))+1;
			int ybar = -1, ybar_bid = -1;
			for(long j = 0; j < N; j++) {
				int ytmp = y_neg[rand() % nr_neg];
				double score_ytmp = do_dot_product(Wxi, H+ytmp*k, k);
				int bid = (int)(std::lower_bound(sorted_y_score.begin(), sorted_y_score.begin()+nr_pos, score_ytmp+2.0)
					- sorted_y_score.begin());
				bucket_cnts[bid] += 1;
				if(bucket_sample[bid] < 0) 
					bucket_sample[bid] = ytmp;
				if (bid > y_fake && ybar < 0) {
					ybar = ytmp;
					ybar_bid = bid;
				}
					
			}

			// Recover y_neg such that y_neg == 0,...,nr_labels-1
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			if(bucket_cnts[0] == N) continue;

			double eta = param->lrate/(double)nr_pos;

			memset(hdiff, 0, sizeof(double)*k);
			for(long j = nr_pos; j > 0; j--)
				bucket_cnts[j-1] += bucket_cnts[j];

			// Stochastic Gradient Update 
			double sum_Lj = 0;
			for(long j = ybar_bid-1; j >= 0; j--){
				double Lj = L[(int)(nr_neg*bucket_cnts[j+1]/N)];
				int y = sorted_y_idx[j];
				double *Hy = H+y*k;
				// Update hdiff
				do_axpy(-Lj, Hy, hdiff, k);
				//Update H_y
				do_axpy(eta*Lj, Wxi, Hy, k);
				project_with_len(Hy, k, wsabieC);
				sum_Lj += Lj;
			}

			// Update hdiff
			double *Hybar = H + ybar*k;
			do_axpy(sum_Lj, Hybar, hdiff, k);

			// Update H_ybar
			do_axpy(-eta*sum_Lj, Wxi, Hybar, k);
			project_with_len(Hybar, k, wsabieC);

#ifdef EXP_DOALL_GALOIS
                        Galois::do_all(boost::counting_iterator<long>(X.row_ptr[i]), boost::counting_iterator<long>(X.row_ptr[i+1]),
                            [&](long idx) {
#else
#pragma omp parallel for schedule(static,50) shared(X,hdiff)
			for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
#endif
				double *Ws = W+X.col_idx[idx]*k;
				for(int t = 0; t < k; t++) {
					Ws[t] -= eta*X.val_t[idx]*hdiff[t];
				}
				project_with_len(Ws, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
                        });
#else
			}
#endif

		}
	}
	free(hdiff);
	free(Wxi);
	free(L);
}

void wsabie_updates_new3(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long nr_samples){
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld  k = %ld\n", nr_samples, k);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	double *Wxi = MALLOC(double, k);
	double *hdiff = MALLOC(double, k);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	// Projection of W and H
	double wsabieC = get_wsabieC(param);
	//wsabie_model_projection(W, H, nr_feats, nr_labels, k, wsabieC);

	vector<int> y_neg(nr_labels);
	vector<int> y_pos(nr_labels);
	vector<int> sorted_y_idx(nr_labels);
	vector<int> bucket_cnts(nr_labels+1);
	vector<int> bucket_sample(nr_labels+1);
	vector<double> sorted_y_score(nr_labels);

	for(long j = 0; j < nr_labels; j++)
		y_neg[j] = (int)j;

	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = rand() % (int)nr_insts; // selected instance
		long nr_pos = Y.nnz_of_row(int(i));
		long nr_neg = nr_labels - nr_pos;
		if (nr_pos == 0)  continue;

		for(int iter = 0; iter < 1; iter++) {
			bucket_cnts[nr_pos] = 0;
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				long j = idx - Y.row_ptr[i];
				int y = Y.col_idx[idx];
				y_pos[j] = y;
				bucket_cnts[j] = 0;
				bucket_sample[j] = -1;
				sorted_y_idx[j] = (int)j;
				sorted_y_score[j] = (j==0)? cal_score(X,W,H,k,i,y,Wxi): do_dot_product(Wxi,H+y*k,k);
			}
			// construct sorted_y_score and sorted_y_idx
			std::stable_sort(sorted_y_idx.begin(), sorted_y_idx.begin() + nr_pos, ScoreComp(&sorted_y_score[0]));
			std::stable_sort(sorted_y_score.begin(), sorted_y_score.begin() + nr_pos);
			for(long j = 0; j < nr_pos; j++)
				sorted_y_idx[j] = Y.col_idx[Y.row_ptr[i]+sorted_y_idx[j]];

			std::stable_sort(y_pos.begin(), y_pos.begin() + nr_pos);
			// Construct y_neg such that all negative indices are in the front of y_neg 
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			// Sampling negative labels for N trials
			//int N = 3*nr_pos;
			//int N = nr_labels - 1;
			int N = (int)std::min((double)nr_labels-1.0, (double)nr_pos*(3+log2(nr_pos)))+1;
			for(long j = 0; j < N; j++) {
				int ytmp = y_neg[rand() % nr_neg];
				double score_ytmp = do_dot_product(Wxi, H+ytmp*k, k);
				int bid = (int)(std::lower_bound(sorted_y_score.begin(), sorted_y_score.begin()+nr_pos, score_ytmp+2.0)
					- sorted_y_score.begin());
				bucket_cnts[bid] += 1;
				if(bucket_sample[bid] < 0) 
					bucket_sample[bid] = ytmp;
			}

			// Recover y_neg such that y_neg == 0,...,nr_labels-1
			for(long j = 1; j <= nr_pos; j++)
				std::swap(y_neg[y_pos[nr_pos-j]], y_neg[nr_labels-j]);

			if(bucket_cnts[0] == N) continue;

			double eta = param->lrate/(double)nr_pos;

			int ybar = -1, ybar_bid = -1;
			int weighted_N = 0, rand_int;
			for(long j = nr_pos; j > 0; j--)
				weighted_N += (int)j*bucket_cnts[j]; 

			rand_int = rand() % weighted_N;
			weighted_N = 0;
			for(long j = nr_pos; j > 0; j--) {
				if(ybar < 0) {
					weighted_N += (int)j*bucket_cnts[j];
					if(rand_int < weighted_N) {
						ybar = bucket_sample[j];
						ybar_bid = (int)j;
					}
				}
				bucket_cnts[j-1] += bucket_cnts[j];
			}

			// Stochastic Gradient Update 
			double sum_Lj = log2(1+ weighted_N*nr_neg/N);
			// Update hdiff
			double *Hybar = H + ybar*k;
			memset(hdiff, 0, sizeof(double)*k);
			do_axpy(ybar_bid*sum_Lj, Hybar, hdiff, k);

			// Update H_ybar
			do_axpy(-eta*ybar_bid*sum_Lj, Wxi, Hybar, k);
			project_with_len(Hybar, k, wsabieC);

			for(long j = ybar_bid-1; j >= 0; j--){
				int y = sorted_y_idx[j];
				double *Hy = H+y*k;
				// Update hdiff
				do_axpy(-sum_Lj, Hy, hdiff, k);
				//Update H_y
				do_axpy(eta*sum_Lj, Wxi, Hy, k);
				project_with_len(Hy, k, wsabieC);
			}

#ifdef EXP_DOALL_GALOIS
                        Galois::do_all(boost::counting_iterator<long>(X.row_ptr[i]), boost::counting_iterator<long>(X.row_ptr[i+1]),
                            [&](long idx) {
#else
#pragma omp parallel for schedule(static,50) shared(X,hdiff)
			for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
#endif
				double *Ws = W+X.col_idx[idx]*k;
				for(int t = 0; t < k; t++) {
					Ws[t] -= eta*X.val_t[idx]*hdiff[t];
				}
				project_with_len(Ws, k, wsabieC);
#ifdef EXP_DOALL_GALOIS
                        });
#else
			}
#endif
		}
	}
	free(hdiff);
	free(Wxi);
	free(L);
}

static double smat_dot_product(smat_t &X, size_t row, double *w) {
	double ret = 0.0;
	for(long idx = X.row_ptr[row]; idx != X.row_ptr[row+1]; idx++) 
		ret += w[X.col_idx[idx]] * X.val_t[idx];
	return ret;
}
static void smat_add_to(double a, smat_t &X, size_t row, double *w) {
	for(long idx = X.row_ptr[row]; idx != X.row_ptr[row+1]; idx++) 
		w[X.col_idx[idx]] += a*X.val_t[idx];
}

void wsabie_updates_3(multilabel_problem *prob, multilabel_parameter *param, double *H, long nr_samples){
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld\n", nr_samples);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	// Projection of W and H
	double wsabieC = get_wsabieC(param);
	//wsabie_model_projection(W, H, nr_feats, nr_labels, k, wsabieC);


	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = rand() % (int)nr_insts;
		if (Y.nnz_of_row(int(i)) == 0)  continue;
		int y = Y.col_idx[Y.row_ptr[i] + rand() % Y.nnz_of_row((int)i)], ybar;
		double score_y = smat_dot_product(X, i, H+y*k) , score_ybar = 0.0, L_tri = -1;

		// Estimate Approximate Rank: N
		int N = 0;
		while (1) {
			N++;
			ybar = rand() % (int)(nr_labels-1);
			if(ybar == y) ybar = (int)nr_labels - 1;
			score_ybar = smat_dot_product(X, i, H+ybar*k);
			L_tri = 1 - score_y + score_ybar;
			if(L_tri > 0 or (N >= (nr_labels-1))) break;
		}

		if(L_tri > 0) {
			double eta = param->lrate;
			double steplength = L[(nr_labels-1)/N]*eta;
			//printf("y %d:%g ybar %d:%g step %g N %d\n", y, score_y, ybar, score_ybar, steplength, N);
			double *Hy = H+y*k, *Hybar = H+ybar*k;

			// Update H_ybar
			smat_add_to(-steplength, X, i, Hybar);
			// Update H_y
			smat_add_to(steplength, X, i, Hy);
			// Projection
			double len;
			len = sqrt(do_dot_product(Hybar, Hybar, k));
			if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hybar, Hybar, k);
			len = sqrt(do_dot_product(Hy, Hy, k));
			if(len > wsabieC) do_axpy(wsabieC/len-1.0, Hy, Hy, k);
		}
	}
	free(L);
	printf("norm %g\n", norm(H, k*nr_labels));
}

void wsabie_updates_2(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H, long *perm, int t0, long nr_samples){
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	double lambda = 1.0/(2*param->Cp);

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld\n", nr_samples);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	double *Wxi = MALLOC(double, k);
	double *hdiff = MALLOC(double, k);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	//permutation
	for(long i = 0; i < nr_insts; i++) {
		long j = i + rand()%(nr_insts - i);
		long tmp = perm[j];
		perm[j] = perm[i];
		perm[i] = tmp;
	}

	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = perm[sample % nr_insts];
		if (Y.nnz_of_row(int(i)) == 0)  continue;
		int y = Y.col_idx[Y.row_ptr[i] + rand() % Y.nnz_of_row((int)i)], ybar;
		double score_y = cal_score(X, W, H, k, i, y, Wxi), score_ybar = 0.0, L_tri = -1;

		double eta = param->lrate; ///(1+param->lrate*lambda * (t0+sample));
		//printf("eta %g\n", eta);
		// Estimate Approximate Rank: N
		int N = 0;
		while (1) {
			N++;
			ybar = rand() % (int)(nr_labels-1);
			if(ybar == y) ybar = (int)nr_labels - 1;
			score_ybar = do_dot_product(Wxi, H+ybar*k, k);
			L_tri = 1 - score_y + score_ybar;
			if(L_tri > 0 or (N >= (nr_labels-1))) break;
		}

		if(L_tri > 0) {
			//double eta = param->lrate;
			double steplength = L[(nr_labels-1)/N]*eta;
			double *Hy = H+y*k, *Hybar = H+ybar*k;
			// Update W
			for(int t = 0; t < k; t++) 
				hdiff[t] = steplength*(Hybar[t]-Hy[t]);
#ifdef EXP_DOALL_GALOIS
                        Galois::do_all(boost::counting_iterator<long>(X.row_ptr[i]), boost::counting_iterator<long>(X.row_ptr[i+1]),
                            [&](long idx) {
#else
#pragma omp parallel for schedule(static,50) shared(X,hdiff)
			for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
#endif
				double *Ws = W+X.col_idx[idx]*k;
				for(int t = 0; t < k; t++) {
					Ws[t] *= (1-lambda*eta);
					Ws[t] -= X.val_t[idx]*hdiff[t];
				}
#ifdef EXP_DOALL_GALOIS
                        });
#else
			}
#endif

			// Update H_ybar
			do_axpy(-lambda*eta, Hybar, Hybar, k);
			do_axpy(-steplength, Wxi, Hybar, k);
			// Update H_y
			do_axpy(-lambda*eta, Hy, Hy, k);
			do_axpy(steplength, Wxi, Hy, k);
		}
	}
	free(hdiff);
	free(Wxi);
	free(L);
}

void wsabie_updates_4(multilabel_problem *prob, multilabel_parameter *param, double *H, long *perm, int t0, long nr_samples){
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long k = param->k; // rank
	long nr_labels = Y.cols;
	long nr_insts = X.rows;

	double lambda = 1.0/(2*param->Cp);

	if(nr_samples <= 0)
		nr_samples = nr_insts;

	printf("GG: nr_samples =%ld\n", nr_samples);

	// L(*) in Eq(5) with  \alpha_j = 1/j
	double *L = MALLOC(double, nr_labels);
	L[0] = 1.0;
	for(int j = 1; j < nr_labels; j++) 
		L[j] = L[j-1] + 1.0 / (1 + j); 

	//permutation
	for(long i = 0; i < nr_insts; i++) {
		long j = i + rand()%(nr_insts - i);
		long tmp = perm[j];
		perm[j] = perm[i];
		perm[i] = tmp;
	}

	// SGD
	for(long sample = 0; sample < nr_samples; sample++) {
		long i = perm[sample % nr_insts];
		if (Y.nnz_of_row(int(i)) == 0)  continue;
		int y = Y.col_idx[Y.row_ptr[i] + rand() % Y.nnz_of_row((int)i)], ybar;
		//double score_y = cal_score(X, W, H, k, i, y, Wxi), score_ybar = 0.0, L_tri = -1;
		double score_y = smat_dot_product(X, i, H+y*k) , score_ybar = 0.0, L_tri = -1;

		double eta = param->lrate; ///(1+param->lrate*lambda * (t0+sample));
		//printf("eta %g\n", eta);
		// Estimate Approximate Rank: N
		int N = 0;
		while (1) {
			N++;
			ybar = rand() % (int)(nr_labels-1);
			if(ybar == y) ybar = (int)nr_labels - 1;
			//score_ybar = do_dot_product(Wxi, H+ybar*k, k);
			score_ybar = smat_dot_product(X, i, H+ybar*k);
			L_tri = 1 - score_y + score_ybar;
			if(L_tri > 0 or (N >= (nr_labels-1))) break;
		}

		if(L_tri > 0) {
			//double eta = param->lrate;
			double steplength = L[(nr_labels-1)/N]*eta;
			double *Hy = H+y*k, *Hybar = H+ybar*k;

			// Update H_ybar
			do_axpy(-lambda*eta, Hybar, Hybar, k);
			smat_add_to(-steplength, X, i, Hybar);
			//do_axpy(-steplength, Wxi, Hybar, k);
			// Update H_y
			do_axpy(-lambda*eta, Hy, Hy, k);
			smat_add_to(steplength, X, i, Hy);
			//do_axpy(steplength, Wxi, Hy, k);
		}
	}
	free(L);
}
