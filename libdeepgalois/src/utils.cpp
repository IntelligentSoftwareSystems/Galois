#include "galois/Galois.h"
#include "deepgalois/utils.h"

namespace deepgalois {

#define NUM_DATASETS 8
const std::string dataset_names[NUM_DATASETS] = {"cora", "citeseer", "ppi", "pubmed", "flickr", "yelp", "reddit", "amazon"};

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
acc_t masked_f1_score(size_t begin, size_t end, size_t count, mask_t *masks, 
                      size_t num_classes, label_t *ground_truth, float_t *pred) {
  double precision_cls(0.), recall_cls(0.), f1_accum(0.);
  int tp_accum(0), fn_accum(0), fp_accum(0), tn_accum(0);
  for (size_t col = 0; col < num_classes; col++) {
    int tp_cls(0), fp_cls(0), fn_cls(0), tn_cls(0);
    for (size_t row = begin; row < end; row ++) {
    //galois::do_all(galois::iterate(begin, end), [&](const auto& row) {
      if (masks[row] == 1) {
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

#ifndef GALOIS_USE_DIST
//! Get masks from datafile where first line tells range of
//! set to create mask from
size_t read_masks(std::string dataset_str, std::string mask_type,
                  size_t n, size_t& begin, size_t& end, mask_t* masks) {
  bool dataset_found = false;
  for (int i = 0; i < NUM_DATASETS; i++) {
    if (dataset_str == dataset_names[i]) {
      dataset_found = true;
      break;
    }
  }
  if (!dataset_found) {
    std::cout << "Dataset currently not supported\n";
    exit(1);
  }
  size_t i             = 0;
  size_t sample_count  = 0;
  std::string filename = path + dataset_str + "-" + mask_type + "_mask.txt";
  // std::cout << "Reading " << filename << "\n";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  in >> begin >> end >> std::ws;
  while (std::getline(in, line)) {
    std::istringstream mask_stream(line);
    if (i >= begin && i < end) {
      unsigned mask = 0;
      mask_stream >> mask;
      if (mask == 1) {
        masks[i] = 1;
        sample_count++;
      }
    }
    i++;
  }
  std::cout << mask_type + "_mask range: [" << begin << ", " << end
    << ") Number of valid samples: " << sample_count << " (" 
    << (float)sample_count/(float)n*(float)100 << "\%)\n";
  in.close();
  return sample_count;
}
#else
size_t read_masks(std::string dataset_str, std::string mask_type,
                         size_t n, size_t& begin, size_t& end,
                         mask_t* masks, Graph* dGraph) {
  bool dataset_found = false;
  for (int i = 0; i < NUM_DATASETS; i++) {
    if (dataset_str == dataset_names[i]) {
      dataset_found = true;
      break;
    }
  }
  if (!dataset_found) {
    std::cout << "Dataset currently not supported\n";
    exit(1);
  }
  size_t i             = 0;
  size_t sample_count  = 0;
  std::string filename = path + dataset_str + "-" + mask_type + "_mask.txt";

  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  in >> begin >> end >> std::ws;
  while (std::getline(in, line)) {
    std::istringstream mask_stream(line);
    if (i >= begin && i < end) {
      unsigned mask = 0;
      mask_stream >> mask;
      if (mask == 1) {
        // only bother if it's local
        if (dGraph->isLocal(i)) {
          masks[dGraph->getLID(i)] = 1;
          sample_count++;
        }
      }
    }
    i++;
  }
  std::cout << mask_type + "_mask range: [" << begin << ", " << end
    << ") Number of valid samples: " << sample_count << "("
    << (float)sample_count/(float)n*(float)100 << "\%)\n";
  in.close();
  return sample_count;
}
#endif

}
