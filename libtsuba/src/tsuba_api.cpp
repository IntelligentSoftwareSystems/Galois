/**
 * @file tsuba_api.cpp
 *
 * Contains the entry points for interfacing with the tsuba storage server
 */

#include <aws/core/Aws.h>
#include <mutex>

#include "tsuba/tsuba.h"
#include "s3.h"
#include "tsuba_internal.h"

static Aws::SDKOptions sdk_options;

void tsuba_init () {
  static std::once_flag init_flag;

  std::call_once(init_flag, [] () {
     Aws::InitAPI(sdk_options);
     std::atexit([] () {
       Aws::ShutdownAPI(sdk_options);
     });
  });
}

EXPORT_SYM int tsuba_open(const char* uri) {
  tsuba_init();

  char name_buf[URI_LIM] = {0};
  char *bucket_name, *object_name;
  if (s3_uri_read(uri, name_buf, &bucket_name, &object_name))
    return ERRNO_RET(EINVAL, -1);

  return s3_open(bucket_name, object_name);
}
