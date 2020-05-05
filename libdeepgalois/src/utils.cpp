#include "galois/Galois.h"
#include "deepgalois/utils.h"

namespace deepgalois {

// parallel prefix sum
template <typename InTy, typename OutTy>
OutTy* parallel_prefix_sum(const std::vector<InTy> &in) {
  const size_t block_size = 1<<20;
  const size_t num_blocks = (in.size() + block_size - 1) / block_size;
  std::vector<OutTy> local_sums(num_blocks);
  // count how many bits are set on each thread
  galois::do_all(galois::iterate((size_t)0, num_blocks), [&](const size_t& block) {
    OutTy lsum = 0;
    size_t block_end = std::min((block + 1) * block_size, in.size());
    for (size_t i=block * block_size; i < block_end; i++)
      lsum += in[i];
    local_sums[block] = lsum;
  });
  std::vector<OutTy> bulk_prefix(num_blocks+1);
  OutTy total = 0;
  for (size_t block=0; block < num_blocks; block++) {
    bulk_prefix[block] = total;
    total += local_sums[block];
  }
  bulk_prefix[num_blocks] = total;
  OutTy *prefix = new OutTy[in.size() + 1];
  galois::do_all(galois::iterate((size_t)0, num_blocks), [&](const size_t& block) {
    OutTy local_total = bulk_prefix[block];
    size_t block_end = std::min((block + 1) * block_size, in.size());
    for (size_t i=block * block_size; i < block_end; i++) {
      prefix[i] = local_total;
      local_total += in[i];
    }
  });
  prefix[in.size()] = bulk_prefix[num_blocks];
  return prefix;
}

template uint32_t* parallel_prefix_sum<uint32_t, uint32_t>(const std::vector<uint32_t> &in);

// Compute the F1 score, also known as balanced F-score or F-measure
// The F1 score can be interpreted as a weighted average of the precision and recall, 
// where an F1 score reaches its best value at 1 and worst score at 0. 
// The relative contribution of precision and recall to the F1 score are equal.
// The formula for the F1 score is:
// F1 = 2 * (precision * recall) / (precision + recall)
// where precision = TP / (TP + FP), recall = TP / (TP + FN)
// TP: true positive; FP: false positive; FN: false negative.
// In the multi-class and multi-label case, this is the weighted average of the F1 score of each class.
// Please refer to https://sebastianraschka.com/faq/docs/multiclass-metric.html,
// http://pageperso.lif.univ-mrs.fr/~francois.denis/IAAM1/scikit-learn-docs.pdf (p.1672)
// and https://github.com/ashokpant/accuracy-evaluation-cpp/blob/master/src/evaluation.hpp
acc_t masked_f1_score(size_t begin, size_t end, size_t, mask_t *masks, 
                      size_t num_classes, label_t *ground_truth, float_t *pred) {
  double precision_cls(0.), recall_cls(0.), f1_accum(0.);
  int tp_accum(0), fn_accum(0), fp_accum(0), tn_accum(0);
  for (size_t col = 0; col < num_classes; col++) {
    int tp_cls(0), fp_cls(0), fn_cls(0), tn_cls(0);
    for (size_t row = begin; row < end; row ++) {
    //galois::do_all(galois::iterate(begin, end), [&](const auto& row) {
      if (masks == NULL || masks[row] == 1) {
        auto idx = row * num_classes + col;
        if (ground_truth[idx] == 1 && pred[idx] > 0.5) {
          //__sync_fetch_and_add(&tp_cls, 1);
          tp_cls += 1;
        } else if (ground_truth[idx] == 0 && pred[idx] > 0.5) {
          //__sync_fetch_and_add(&fp_cls, 1);
          fp_cls += 1;
        } else if (ground_truth[idx] == 1 && pred[idx] <= 0.5) {
          //__sync_fetch_and_add(&fn_cls, 1);
          fn_cls += 1;
        } else if (ground_truth[idx] == 0 && pred[idx] <= 0.5) {
          //__sync_fetch_and_add(&tn_cls, 1);
          tn_cls += 1;
        }
      }
    }
    //}, galois::loopname("MaskedF1Score"));
    tp_accum += tp_cls;
    fn_accum += fn_cls;
    fp_accum += fp_cls;
    tn_accum += tn_cls;
    precision_cls = tp_cls + fp_cls > 0 ? (double)tp_cls/(double)(tp_cls+fp_cls) : 0.;
    recall_cls = tp_cls+fn_cls > 0 ? (double)tp_cls/(double)(tp_cls+fn_cls) : 0.;
    f1_accum += recall_cls+precision_cls > 0. ? 2.*(recall_cls*precision_cls)/(recall_cls+precision_cls) : 0.;
  }
  double f1_macro = f1_accum/(double)num_classes;
  //double accuracy_mic = (double)(tp_accum+tn_accum)/(double)(tp_accum+tn_accum+fp_accum+fn_accum);
  double precision_mic = tp_accum+fp_accum > 0 ? (double)tp_accum/(double)(tp_accum+fp_accum) : 0.;
  double recall_mic = tp_accum+fn_accum > 0 ? (double)tp_accum/(double)(tp_accum+fn_accum) : 0.;
  double f1_micro = recall_mic+precision_mic > 0. ? 2.*(recall_mic*precision_mic)/(recall_mic+precision_mic) : 0.;
  std::cout << std::setprecision(3) << std::fixed <<
      " (f1_micro: " << f1_micro << ", f1_macro: " << f1_macro << ") ";
  return f1_micro;
}

} // end namespace
