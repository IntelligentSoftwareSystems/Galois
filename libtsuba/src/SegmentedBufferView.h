#ifndef GALOIS_LIBTSUBA_SEGMENTED_BUFFER_VIEW_H_
#define GALOIS_LIBTSUBA_SEGMENTED_BUFFER_VIEW_H_

#include <iterator>
#include <algorithm>
#include <cstdint>

namespace tsuba {

/* generate pointers to aligned, step_-sized parts of a buffer
 */
class SegmentedBufferView {
  uint64_t start_;
  uint8_t* raw_;
  uint64_t size_;
  uint64_t step_;

public:
  class iterator;
  struct BufPart {
    uint64_t start;
    uint64_t end;
    uint8_t* dest;
  };

  SegmentedBufferView(uint64_t start, uint8_t* raw, uint64_t size,
                      uint64_t step)
      : start_(start), raw_(raw), size_(size), step_(step) {}
  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, size_); }

  class iterator {
    SegmentedBufferView* buf_view_;
    uint64_t offset_;
    uint64_t next_step_;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type        = BufPart;
    using difference_type   = int64_t;
    using pointer           = BufPart*;
    using reference         = BufPart&;
    iterator(SegmentedBufferView* buf_view, uint64_t offset)
        : buf_view_(buf_view), offset_(offset) {
      if (buf_view_->size_ <= buf_view_->step_) {
        next_step_ = buf_view_->size_;
      } else {
        next_step_ = std::min(buf_view_->step_, buf_view_->size_ - offset_);
      }
    }
    BufPart operator*() {
      return BufPart{
          .start = buf_view_->start_ + offset_,
          .end   = buf_view_->start_ + offset_ + next_step_,
          .dest  = buf_view_->raw_ + offset_, /* NOLINT */
      };
    }
    iterator& operator++() {
      offset_ += next_step_;
      next_step_ = std::min(buf_view_->step_, buf_view_->size_ - offset_);
      return *this;
    }
    bool operator!=(const iterator& other) const {
      return offset_ != other.offset_;
    }
  };
};

} /* namespace tsuba */

#endif
