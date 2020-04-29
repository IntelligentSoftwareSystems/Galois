#ifndef GALOIS_LIBTSUBA_S3_H_
#define GALOIS_LIBTSUBA_S3_H_

#include <string>
#include <cstdint>

namespace tsuba {

int S3Init();
void S3Fini();
int S3Open(const std::string& bucket, const std::string& object);
uint64_t S3GetSize(const std::string& bucket, const std::string& object,
                   uint64_t* size);

std::pair<std::string, std::string> S3SplitUri(const std::string& uri);

int S3DownloadRange(const std::string& bucket, const std::string& object,
                    uint64_t start, uint64_t size, uint8_t* result_buf);

int S3UploadOverwrite(const std::string& bucket, const std::string& object,
                      const uint8_t* data, uint64_t size);
} /* namespace tsuba */

#endif
