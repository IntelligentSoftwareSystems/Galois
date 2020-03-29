#ifndef GALOIS_LIBGALOIS_GALOIS_FILE_VIEW_H_
#define GALOIS_LIBGALOIS_GALOIS_FILE_VIEW_H_

#include <string>
#include <cstdint>

namespace galois {

class FileView {
  uint8_t* map_start_;
  uint8_t* region_start_;
  uint64_t map_size_;
  uint64_t region_size_;
  bool valid_ = false;

public:
  FileView()                = default;
  FileView(const FileView&) = delete;
  FileView& operator=(const FileView&) = delete;

  FileView(FileView&& other) noexcept
      : map_start_(other.map_start_), region_start_(other.region_start_),
        map_size_(other.map_size_), region_size_(other.region_size_),
        valid_(other.valid_) {
    other.valid_ = false;
  }

  FileView& operator=(FileView&& other) noexcept {
    if (&other != this) {
      Unbind();
      map_start_    = other.map_start_;
      region_start_ = other.region_start_;
      map_size_     = other.map_size_;
      region_size_  = other.region_size_;
      valid_        = other.valid_;
      other.valid_  = false;
    }
    return *this;
  }

  ~FileView();

  int Bind(const std::string& filename, uint64_t begin, uint64_t end);
  inline int Bind(const std::string& filename, uint64_t stop) {
    return Bind(filename, 0, stop);
  }
  void Unbind();

  template <typename T>
  const T* ptr() const {
    return reinterpret_cast<T*>(region_start_); /* NOLINT */
  }

  uint64_t size() const { return region_size_; }
};

} /* namespace galois */

#endif
