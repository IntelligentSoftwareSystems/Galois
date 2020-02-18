#include <memory>
#include <mutex>
#include <unistd.h>

#include <aws/s3/S3Client.h>
#include <aws/transfer/TransferManager.h>
#include <aws/core/utils/threading/Executor.h>

#include "s3.h"
#include "tsuba_internal.h"
#include "tsuba/tsuba.h"

int s3_uri_read(const char *uri, char *buf, char **bucket, char **object) {

  if (!tsuba_is_uri(uri))
    return ERRNO_RET(EINVAL, -1);

  strncpy(buf, uri + 5, URI_LIM);

  /* null terminate at the first slash to separate the bucket and the object */
  char* slash = strchr(buf, '/');
  if (!slash)
    return -1;

  *slash = '\0';

  *bucket = buf;
  *object = slash + 1;
  return 0;
}

int s3_open(const char *bucket, const char *object) {
  auto s3_client = Aws::MakeShared<Aws::S3::S3Client>("TsubaS3Client");
  auto executor  = Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
      "executor", n_cpus());

  Aws::Transfer::TransferManagerConfiguration transfer_config(executor.get());
  transfer_config.s3Client = s3_client;
  auto transfer_manager =
      Aws::Transfer::TransferManager::Create(transfer_config);

  char tmpname[] = "/tmp/tsuba_s3.XXXXXX";
  int fd = mkstemp(tmpname);
  auto downloadHandle =
      transfer_manager->DownloadFile(bucket, object, tmpname);
  downloadHandle->WaitUntilFinished();

  assert(downloadHandle->GetBytesTotalSize() ==
         downloadHandle->GetBytesTransferred());
  unlink(tmpname);
  return fd;
}
