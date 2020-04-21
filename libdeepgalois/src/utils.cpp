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
// TP: true positive; FP: false positive; FN: false negtive.
// In the multi-class and multi-label case, this is the weighted average of the F1 score of each class.
// Please refer to https://sebastianraschka.com/faq/docs/multiclass-metric.html,
// http://pageperso.lif.univ-mrs.fr/~francois.denis/IAAM1/scikit-learn-docs.pdf (p.1672)
// and https://github.com/ashokpant/accuracy-evaluation-cpp/blob/master/src/evaluation.hpp
acc_t masked_f1_score(size_t begin, size_t end, size_t count, mask_t *masks, 
                      size_t num_classes, label_t *ground_truth, float_t *pred) {
  float beta = 1.0;
  std::vector<int> true_positive(num_classes, 0);
  std::vector<int> false_positive(num_classes, 0);
  std::vector<int> false_negtive(num_classes, 0);
  galois::do_all(galois::iterate(begin, end), [&](const auto& i) {
    if (masks[i] == 1) {
      for (size_t j = 0; j < num_classes; j++) {
        auto idx = i * num_classes + j;
        if (ground_truth[idx] == 1 && pred[idx] > 0.5) {
          __sync_fetch_and_add(&true_positive[j], 1);
        } else if (ground_truth[idx] == 0 && pred[idx] > 0.5) {
          __sync_fetch_and_add(&false_positive[j], 1);
        } else if (ground_truth[idx] == 1 && pred[idx] <= 0.5) {
          __sync_fetch_and_add(&false_negtive[j], 1);
        }
      }
	}
  }, galois::loopname("MaskedF1Score"));
  acc_t pNumerator = 0.0;
  acc_t pDenominator = 0.0;
  acc_t rNumerator = 0.0;
  acc_t rDenominator = 0.0;
  for (size_t i = 0; i < num_classes; i++) {
    acc_t fn = (acc_t)false_negtive[i]; // false negtive
    acc_t fp = (acc_t)false_positive[i]; // false positive
	acc_t tp = (acc_t)true_positive[i]; // true positive
	pNumerator = pNumerator + tp;
	pDenominator = pDenominator + (tp + fp);
    rNumerator = rNumerator + tp;
    rDenominator = rDenominator + (tp + fn);
  }
  auto recallMicro = rNumerator / rDenominator;
  acc_t precisionMicro = pNumerator / pDenominator;
  auto fscoreMicro = (((beta * beta) + 1) * precisionMicro * recallMicro) / 
                     ((beta * beta) * precisionMicro + recallMicro);
  return fscoreMicro;
}

#ifndef GALOIS_USE_DIST
//! Get masks from datafile where first line tells range of
//! set to create mask from
size_t read_masks(std::string dataset_str, std::string mask_type,
                         size_t& begin, size_t& end, std::vector<uint8_t>& masks) {
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
    << (float)sample_count/(float)masks.size()*(float)100 << "\%)\n";
  in.close();
  return sample_count;
}
#else
size_t read_masks(std::string dataset_str, std::string mask_type,
                         size_t& begin, size_t& end,
                         std::vector<uint8_t>& masks, Graph* dGraph) {
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
    << ") Number of valid samples: " << sample_count << "\n";
  in.close();
  return sample_count;
}
#endif

}
