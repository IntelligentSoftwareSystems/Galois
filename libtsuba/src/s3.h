#ifndef GALOIS_LIBTSUBA_S3_H_
#define GALOIS_LIBTSUBA_S3_H_

#include <string>
#include <cstdint>

namespace tsuba {

int S3Open(const std::string& bucket, const std::string& object);

std::pair<std::string, std::string> S3SplitUri(const std::string& /*uri*/);

int S3DownloadRange(const std::string& bucket, const std::string& object,
                    uint64_t start, uint64_t size, uint8_t* result_buf);
} /* namespace tsuba */

#endif
