#ifndef GALOIS_LARGEVECTOR_H
#define GALOIS_LARGEVECTOR_H

#include <linux/memfd.h>
#include <linux/mman.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <iostream>
#include <list>
#include <string.h>
#include <utility>
namespace galois {
/*
 * A vector backed by huge pages. Guarantees addresss stability, so values do
 * not have to be moveable.
 *
 * A note on iterator safety:
 *  1. Resizing the container results in a new iterator generation.
 *  2. All iterator methods (e.g. increment) preserve generation.
 *  3. It is undefined behavior to compare iterators across generations.
 *  4. Decreasing the container size invalidates some iterators.
 */
template <typename T>
class LargeVector : public boost::noncopyable {
private:
  size_t m_capacity;
  size_t m_size;
  T* m_data;

  int m_fd;
  std::list<std::pair<void*, size_t>> m_mappings;

  void ensure_capacity(size_t new_cap) {
    using std::string;

    if (m_capacity == new_cap)
      return;

    // Round up to the nearest huge page size.
    constexpr size_t page_size = 1ull << 21;
    const size_t file_size =
        (new_cap * sizeof(T) + (page_size - 1)) & (~(page_size - 1));

    if (ftruncate(m_fd, file_size) == -1)
      throw std::runtime_error(string("ftruncate: ") + strerror(errno));

    // Floor divide to find the real capacity.
    new_cap = file_size / sizeof(T);

    // We only need to remap if the new capacity is larger. Otherwise,
    // we'll just truncate the file to release any used physical pages
    // and keep the existing mapping.
    const bool remap = new_cap > m_capacity;

    m_capacity = new_cap;

    if (!remap)
      return;

    const size_t mmap_size = m_capacity * sizeof(T);
    if (!mmap_size) {
      m_data = nullptr;
      return;
    }

    m_data =
        static_cast<T*>(mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_HUGETLB | MAP_HUGE_2MB, m_fd, 0));
    if (m_data == MAP_FAILED)
      throw std::runtime_error(string("mmap failed: ") + strerror(errno));

    m_mappings.push_back(std::make_pair(m_data, mmap_size));
  }

public:
  LargeVector(size_t initial_capacity)
      : m_capacity(0), m_size(0), m_data(nullptr),
        m_fd(memfd_create("LargeVector", MFD_HUGETLB | MFD_HUGE_2MB)) {
    if (m_fd == -1) {
      throw std::runtime_error(std::string("creating memfd: ") +
                               strerror(errno));
    }
    ensure_capacity(initial_capacity);
  }

  LargeVector() : LargeVector(1) {}

  LargeVector(LargeVector&& other)
      : m_capacity(other.m_capacity), m_size(other.m_size),
        m_data(other.m_data), m_fd(other.m_fd),
        m_mappings(std::move(other.m_mappings)) {
    other.m_capacity = 0;
    other.m_size     = 0;
    other.m_data     = nullptr;
    other.m_fd       = -1;
    assert(other.m_mappings.empty());
  }

  LargeVector& operator=(LargeVector<T>&& other) {
    m_capacity = std::move(other.m_capacity);
    m_size     = std::move(other.m_size);
    m_data     = std::move(other.m_data);
    m_fd       = std::move(other.m_fd);
    m_mappings = std::move(other.m_mappings);

    other.m_capacity = 0;
    other.m_size     = 0;
    other.m_data     = nullptr;
    other.m_fd       = -1;
    assert(other.m_mappings.empty());

    return *this;
  }

  ~LargeVector() {
    for (; !m_mappings.empty(); m_mappings.pop_front())
      munmap(m_mappings.front().first, m_mappings.front().second);

    if (m_fd != -1)
      close(m_fd);
  }

  uint64_t size() const noexcept { return m_size; }

  template <typename... Args>
  T& emplace_back(Args&&... args) {
    if (m_size == m_capacity) {
      ensure_capacity(m_size + 1);
    }
    return *new (m_data + m_size++) T(std::forward<Args>(args)...);
  }

  T& push_back(const T& t) { return emplace_back(t); }

  T& push_back(T&& t) { return emplace_back(std::move(t)); }

  T& operator[](size_t index) const { return m_data[index]; }

  void pop_back() {
    assert(m_size > 0);
    m_data[--m_size].~T();
  }

  void resize(size_t count) {
    for (T* ii = begin() + count; ii < end(); ++ii)
      ii->~T();

    ensure_capacity(count);

    for (T* ii = end(); ii < begin() + count; ++ii)
      new (ii) T();

    m_size = count;
  }

  inline T* begin() { return m_data; }
  inline T* end() { return m_data + m_size; }
};

}; // namespace galois

namespace std {
template <typename T>
ostream& operator<<(std::ostream& os, const galois::LargeVector<T>& vec) {
  for (uint64_t i = 0; i < vec.getSize(); i++) {
    os << vec[i];
    if (i < vec.getSize() - 1) {
      os << " ";
    }
  }
  return os;
}

template <typename T>
istream& operator>>(istream& is, galois::LargeVector<T>& vec) {
  T value;
  while (is >> value) {
    vec.push_back(value);
  }
  return is;
}
} // namespace std

#endif