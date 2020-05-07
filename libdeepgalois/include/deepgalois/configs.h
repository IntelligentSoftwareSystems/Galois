#pragma once

namespace deepgalois {

const std::string path =
    "/net/ohm/export/iss/inputs/Learning/"; // path to the input dataset

#define NUM_DATASETS 8
const std::string dataset_names[NUM_DATASETS] = {
    "cora", "citeseer", "ppi", "pubmed", "flickr", "yelp", "reddit", "amazon"};

} // namespace deepgalois
