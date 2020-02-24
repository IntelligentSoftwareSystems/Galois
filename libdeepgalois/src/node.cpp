#include "node.h"

void edge::alloc() {
#ifdef CPU_ONLY
	data_ = new float_t[num_samples_ * ft_dim_];
	grad_ = new float_t[num_samples_ * ft_dim_];
#else
	alloc_gpu();
#endif
}

void edge::merge_grads(vec_t *dst) {
	assert(grad_ != NULL);
	dst->resize(ft_dim_);
	float_t *pdst = &(*dst)[0];
#ifdef CPU_ONLY
	std::copy(grad_, grad_+ft_dim_, pdst);
	// @todo consider adding parallelism and vectorization
	for (size_t sample = 1; sample < num_samples_; ++sample) {
		for (size_t i = 0; i < ft_dim_; i++) pdst[i] += grad_[sample*ft_dim_+i];
		//vectorize::reduce<float_t>(&grad_[sample][0], ft_dim_, pdst);
	}
#else
	merge_grads_gpu(pdst);
#endif
}

void edge::clear_grads() {
#ifdef CPU_ONLY
	std::fill(grad_, grad_+ft_dim_*num_samples_, float_t{0}); // TODO: need vectorize
	//vectorize::fill(&grad_[0], grad_.size(), float_t{0});
#else
	clear_grads_gpu();
#endif
}

