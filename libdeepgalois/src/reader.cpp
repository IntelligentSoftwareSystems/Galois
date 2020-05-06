#include "deepgalois/reader.h"
#include "deepgalois/utils.h"
#include "deepgalois/configs.h"

namespace deepgalois {

// labels contain the ground truth (e.g. vertex classes) for each example
// (num_examples x 1). Note that labels is not one-hot encoded vector and it can
// be computed as y.argmax(axis=1) from one-hot encoded vector (y) of labels if
// required.
size_t Reader::read_labels(bool is_single_class, label_t*& labels) {
  std::cout << "Reading labels ... ";
  Timer t_read;
  t_read.Start();
  std::string filename = path + dataset_str + "-labels.txt";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m, num_classes; // m: number of samples
  in >> m >> num_classes >> std::ws;
  if (is_single_class) {
    std::cout << "Using single-class (one-hot) labels\n";
    labels = new label_t[m]; // single-class (one-hot) label for each vertex: N x 1
  } else {
    std::cout << "Using multi-class labels\n";
   labels = new label_t[m*num_classes]; // multi-class label for each vertex: N x E
  }
  unsigned v = 0;
  while (std::getline(in, line)) {
    std::istringstream label_stream(line);
    unsigned x;
    for (size_t idx = 0; idx < num_classes; ++idx) {
      label_stream >> x;
      if (is_single_class) {
        if (x != 0) {
          labels[v] = idx;
          break;
        }
      } else {
        labels[v*num_classes+idx] = x;
      }
    }
    v++;
  }
  in.close();
  t_read.Stop();
  // print the number of vertex classes
  std::cout << "Done, unique label counts: " << num_classes
            << ", time: " << t_read.Millisecs() << " ms\n";
  //for (auto i = 0; i < 10; i ++) std::cout << "labels[" << i << "] = " << unsigned(labels[i]) << "\n";
  return num_classes;
}

//! Read features, return the length of a feature vector
//! Features are stored in the Context class
size_t Reader::read_features(float_t*& feats, std::string filetype) {
  //filetype = "txt";
  std::cout << "Reading features ... ";
  Timer t_read;
  t_read.Start();
  size_t m, feat_len; // m = number of vertices
  std::string filename = path + dataset_str + ".ft";
  std::ifstream in;

  if (filetype == "bin") {
    std::string file_dims = path + dataset_str + "-dims.txt";
    std::ifstream ifs;
    ifs.open(file_dims, std::ios::in);
    ifs >> m >> feat_len >> std::ws;
    ifs.close();
  } else {
    in.open(filename, std::ios::in);
    in >> m >> feat_len >> std::ws;
  }
  std::cout << "N x D: " << m << " x " << feat_len << "\n";
  feats = new float_t[m * feat_len];
  if (filetype == "bin") {
    filename = path + dataset_str + "-feats.bin";
    in.open(filename, std::ios::binary|std::ios::in);
    in.read((char*)feats, sizeof(float_t) * m * feat_len);
  } else {
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream edge_stream(line);
      unsigned u, v;
      float_t w;
      edge_stream >> u;
      edge_stream >> v;
      edge_stream >> w;
      feats[u * feat_len + v] = w;
    }
  }
  in.close();
  t_read.Stop();
  std::cout << "Done, feature length: " << feat_len
            << ", time: " << t_read.Millisecs() << " ms\n";
  //for (auto i = 0; i < 6; i ++) 
    //for (auto j = 0; j < 6; j ++) 
      //std::cout << "feats[" << i << "][" << j << "] = " << feats[i*feat_len+j] << "\n";
  return feat_len;
}

//! Get masks from datafile where first line tells range of
//! set to create mask from
size_t Reader::read_masks(std::string mask_type, size_t n, size_t& begin, size_t& end, mask_t* masks) {
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

}
