namespace galois {
namespace runtime {

namespace internal {
void print_send_impl(std::vector<uint8_t>, size_t, unsigned);
void print_recv_impl(std::vector<uint8_t>, size_t, unsigned);
}

static inline void print_send(std::vector<uint8_t> vec, size_t len, unsigned host){
  internal::print_send_impl(vec, len, host);
}

} // namespace runtime
} // namespace galois

