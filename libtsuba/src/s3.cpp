#include <memory>
#include <regex>
#include <string_view>
#include <algorithm>
#include <unistd.h>

#include <aws/s3/S3Client.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/transfer/TransferManager.h>
#include <aws/core/utils/threading/Executor.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>

#include "tsuba_internal.h"
#include "s3.h"

namespace tsuba {

static constexpr const char* kDefaultS3Region = "us-east-2";
static constexpr const char* kAwsTag          = "TsubaS3Client";
static const std::regex kS3UriRegex("s3://([-a-z0-9.]+)/(.+)");
static const std::string_view kTmpTag("/tmp/tsuba_s3.XXXXXX");
static constexpr const uint64_t kS3BufSize = MB(5);

static inline std::shared_ptr<Aws::S3::S3Client> GetS3Client() {
  Aws::Client::ClientConfiguration cfg;
  cfg.region = kDefaultS3Region;
  return Aws::MakeShared<Aws::S3::S3Client>(kAwsTag, cfg);
}

std::pair<std::string, std::string> S3SplitUri(const std::string& uri) {
  std::smatch sub_match;
  /* I wish regex was compatible with string_view but alas it is not */
  if (!std::regex_match(uri, sub_match, kS3UriRegex)) {
    return std::make_pair("", "");
  }
  return std::make_pair(sub_match[1], sub_match[2]);
}

int S3Open(const std::string& bucket, const std::string& object) {
  Aws::Client::ClientConfiguration cfg;
  cfg.region     = kDefaultS3Region;
  auto s3_client = Aws::MakeShared<Aws::S3::S3Client>(kAwsTag, cfg);
  auto executor =
      Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(kAwsTag, 1);

  Aws::Transfer::TransferManagerConfiguration transfer_config(executor.get());
  transfer_config.s3Client = s3_client;
  auto transfer_manager =
      Aws::Transfer::TransferManager::Create(transfer_config);

  std::vector<char> tmpname(kTmpTag.begin(), kTmpTag.end());
  int fd = mkstemp(tmpname.data());
  auto downloadHandle =
      transfer_manager->DownloadFile(bucket, object, tmpname.data());
  downloadHandle->WaitUntilFinished();

  assert(downloadHandle->GetBytesTotalSize() ==
         downloadHandle->GetBytesTransferred());
  unlink(tmpname.data());
  return fd;
}

struct FilePart {
  uint64_t start;
  uint64_t end;
  uint8_t* dest;
};

int S3DownloadRange(const std::string& bucket, const std::string& object,
                    uint64_t start, uint64_t size, uint8_t* result_buf) {
  auto s3_client = GetS3Client();
  std::vector<FilePart> parts;

  uint64_t last = start + size;
  while (start < last) {
    uint64_t sz = std::min(last - start, kS3BufSize);
    /* Range in AWS S3 API is inclusive */
    uint64_t end = start + sz - 1;
    parts.emplace_back(
        FilePart{.start = start, .end = end, .dest = result_buf});
    start += kS3BufSize;
    result_buf += kS3BufSize; /* NOLINT */
  }

  std::mutex m;
  std::condition_variable cv;

  uint64_t finished = 0;
  auto callback =
      [&](const Aws::S3::S3Client* /*clnt*/,
          const Aws::S3::Model::GetObjectRequest& /*req*/,
          const Aws::S3::Model::GetObjectOutcome& get_object_outcome,
          const std::shared_ptr<
              const Aws::Client::AsyncCallerContext>& /*ctx*/) {
        if (get_object_outcome.IsSuccess()) {
          /* result_buf should have our data here */
          std::unique_lock<std::mutex> lk(m);
          finished++;
          cv.notify_one();
        } else {
          /* TODO there are likely some errors we can handle gracefully
           * i.e., with retries */
          const auto& error = get_object_outcome.GetError();
          std::cout << "ERROR: " << error.GetExceptionName() << ": "
                    << error.GetMessage() << std::endl;
          abort();
        }
      };
  for (auto& part : parts) {
    Aws::S3::Model::GetObjectRequest object_request;
    object_request.SetBucket(bucket);
    object_request.SetKey(object);
    std::ostringstream range;
    range << "bytes=" << part.start << "-" << part.end;
    object_request.SetRange(range.str());

    object_request.SetResponseStreamFactory([&]() {
      auto* bufferStream =
          Aws::New<Aws::Utils::Stream::DefaultUnderlyingStream>(
              kAwsTag,
              Aws::MakeUnique<Aws::Utils::Stream::PreallocatedStreamBuf>(
                  kAwsTag, part.dest, part.end - part.start + 1));
      if (bufferStream == nullptr) {
        abort();
      }
      return bufferStream;
    });
    s3_client->GetObjectAsync(object_request, callback);
  }

  std::unique_lock<std::mutex> lk(m);
  cv.wait(lk, [&] { return finished >= parts.size(); });

  return 0;
}

} /* namespace tsuba */
