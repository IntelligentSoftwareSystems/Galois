/**
 * @file tsuba_api.cpp
 *
 * Contains the entry points for interfacing with the tsuba storage server
 */

#include <aws/core/Aws.h>
#include <mutex>
#include <unordered_map>
#include <fstream>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "tsuba/tsuba.h"
#include "s3.h"
#include "tsuba_internal.h"

static Aws::SDKOptions sdk_options;

namespace tsuba {

template <typename T>
static T* MmapCast(size_t size, int prot, int flags, int fd, off_t off) {
  void* ret = mmap(nullptr, size, prot, flags, fd, off);
  if (ret == MAP_FAILED) {
    perror("mmap");
    return nullptr;
  }
  return reinterpret_cast<T*>(ret); /* NOLINT cast needed for mmap interface */
}

static uint8_t* MmapLocalFile(const char* filename, uint64_t begin,
                              uint64_t size) {
  int fd = open(filename, O_RDONLY, 0);
  if (fd < 0) {
    return nullptr;
  }
  auto* ret = MmapCast<uint8_t>(size, PROT_READ, MAP_SHARED, fd, begin);
  close(fd);
  return ret;
}

static int DoReadS3Part(const char* filename, uint8_t* buf, uint64_t begin,
                        uint64_t size) {
  auto [bucket, object] = S3SplitUri(filename);
  return S3DownloadRange(bucket, object, begin, size, buf);
}

static uint8_t* AllocAndReadS3(const char* filename, uint64_t begin,
                               uint64_t size) {
  auto* ret = MmapCast<uint8_t>(size, PROT_READ | PROT_WRITE,
                                MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (ret == nullptr) {
    std::cerr << "alloc for s3 read failed" << std::endl;
    return nullptr;
  }
  if (DoReadS3Part(filename, ret, begin, size)) {
    munmap(ret, size);
    ret = nullptr;
    std::cerr << "do_read_s3_part failed" << std::endl;
  }
  return ret;
}

class MappingDesc {
  uint8_t* ptr_{nullptr};
  size_t size_{0};
  bool valid_ = false;

public:
  MappingDesc(const MappingDesc&) = delete;
  MappingDesc& operator=(const MappingDesc&) = delete;
  MappingDesc()                              = default;
  MappingDesc(MappingDesc&& other) noexcept
      : ptr_(other.ptr_), size_(other.size_), valid_(other.valid_) {
    other.valid_ = false;
  }
  MappingDesc& operator=(MappingDesc&& other) noexcept {
    if (&other != this) {
      if (valid_) {
        munmap(ptr_, size_);
      }
      ptr_         = other.ptr_;
      size_        = other.size_;
      valid_       = other.valid_;
      other.valid_ = false;
    }
    return *this;
  }
  int init(const std::string& uri, uint64_t offset, size_t size) {
    assert(!valid_);
    if (TsubaIsUri(uri)) {
      ptr_ = AllocAndReadS3(uri.c_str(), offset, size);
    } else {
      ptr_ = MmapLocalFile(uri.c_str(), offset, size);
    }
    if (ptr_ == nullptr) {
      return -1;
    }
    valid_ = true;
    size_  = size;
    return 0;
  }
  ~MappingDesc() {
    if (valid_) {
      munmap(ptr_, size_);
    }
  }
  uint8_t* ptr() { return ptr_; }
};

static std::mutex allocated_memory_mutex;
static std::unordered_map<uint8_t*, MappingDesc> allocated_memory;

EXPORT_SYM int TsubaInit() {
  Aws::InitAPI(sdk_options);
  return 0;
}
EXPORT_SYM void TsubaFini() { Aws::ShutdownAPI(sdk_options); }

EXPORT_SYM int TsubaOpen(const char* uri) {
  auto [bucket_name, object_name] = S3SplitUri(std::string(uri));
  if (bucket_name.empty() || object_name.empty()) {
    return ERRNO_RET(EINVAL, -1);
  }
  return S3Open(bucket_name, object_name);
}

EXPORT_SYM int TsubaPeek(const char* filename, uint8_t* result_buffer,
                         uint64_t begin, uint64_t size) {
  if (!TsubaIsUri(filename)) {
    std::ifstream infile(filename);
    if (!infile.good()) {
      return -1;
    }
    infile.seekg(begin);
    if (!infile.good()) {
      return -1;
    }
    infile.read(reinterpret_cast<char*>(result_buffer), size); /* NOLINT */
    if (!infile.good()) {
      return -1;
    }
    return 0;
  }
  return DoReadS3Part(filename, result_buffer, begin, size);
}

EXPORT_SYM uint8_t* TsubaMmap(const char* filename, uint64_t begin,
                              uint64_t size) {
  MappingDesc new_mapping;
  if (int err = new_mapping.init(filename, begin, size); err) {
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(allocated_memory_mutex);
  auto [it, inserted] =
      allocated_memory.emplace(new_mapping.ptr(), std::move(new_mapping));
  if (!inserted) {
    std::cerr << "Failed to emplace! (impossible?) " __FILE__ ":" << __LINE__
              << std::endl;
    return nullptr;
  }
  return it->second.ptr();
}

EXPORT_SYM void TsubaMunmap(uint8_t* ptr) {
  std::lock_guard<std::mutex> guard(allocated_memory_mutex);
  if (allocated_memory.erase(ptr) != 1) {
    std::cerr << "WARNING: passed unknown pointer to tsuba_munmap" << std::endl;
  }
}

} /* namespace tsuba */
