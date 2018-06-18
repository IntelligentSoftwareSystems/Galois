namespace galois {

class VecBool {
  std::vector<uint8_t> vec;
  uint64_t numItems;
  uint32_t size_each;

public:
  VecBool() = default;
  VecBool(uint64_t _numItems, uint32_t _size_each)
      : numItems(_numItems), size_each(_size_each) {
    vec.resize((_numItems * _size_each + 7) / 8);
  }

  void resize(uint64_t _numItems, uint32_t _size_each) {
    numItems  = _numItems;
    size_each = _size_each;
    vec.resize((_numItems * _size_each + 7) / 8, false);
    std::fill(vec.begin(), vec.end(), 0);
    std::cerr << "resizing : " << vec[0] << "\n";
  }

  bool set_bit_and_return(uint64_t n, uint32_t a) {
    auto addr    = n * size_each + a;
    auto old_val = vec[addr / 8] & (1 << (addr % 8));
    vec[addr / 8] |= (1 << (addr % 8));
    return old_val;
  }

  void set_bit(uint64_t n, uint32_t a) {
    auto addr = n * size_each + a;
    vec[addr / 8] |= (1 << (addr % 8));
  }

  bool is_set(uint64_t n, uint32_t a) const {
    auto addr = n * size_each + a;
    return vec[addr / 8] & (1 << (addr % 8));
  }

  uint64_t size() const { return numItems; }

  uint32_t bit_count(uint64_t n) const {
    uint32_t set_bit_count = 0;
    for (uint32_t k = 0; k < size_each; ++k) {
      if (is_set(n, k)) {
        set_bit_count++;
      }
    }
    return set_bit_count;
  }
  uint32_t find_first(uint64_t n) const {
    for (uint32_t k = 0; k < size_each; ++k) {
      if (is_set(n, k)) {
        return k;
      }
    }
    return ~0;
  }
  uint32_t find_next(uint64_t n, uint32_t p) const {
    for (auto k = p + 1; k < size_each; ++k) {
      if (is_set(n, k)) {
        return k;
      }
    }
    return ~0;
  }

  void clear() { std::vector<uint8_t>().swap(vec); }
};

} // namespace galois
