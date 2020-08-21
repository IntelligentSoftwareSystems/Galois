#ifndef GALOIS_RUNTIME_STACKTRACER_H
#define GALOIS_RUNTIME_STACKTRACER_H
#include <cstdint>

struct MPSCBuffer {
  volatile uint64_t head;
  volatile uintptr_t* buf;
  uint16_t LIMIT;

public:
  /* Old Constructors
  MPSCBuffer(uint16_t lim) : head(0), buf(new uintptr_t[lim]), LIMIT(lim)
  {
    for(uint16_t i = 0; i < LIMIT; i++) buf[i] = 0;
  }

  MPSCBuffer(uint16_t lim, volatile uintptr_t* buf) : head(0), buf(buf),
  LIMIT(lim) {}
  */

  // Will overwrite value if you go over allocated buf size
  int put(uintptr_t val) {
    // Check if input is valid
    if (val == 0)
      return -1;

    uint64_t my_buf_ptr = __atomic_fetch_add(&head, 1, __ATOMIC_RELAXED);
    __atomic_store_8(&buf[my_buf_ptr % LIMIT], val, __ATOMIC_RELAXED);

    return 0;
  }

  uint16_t get_size() {
    uint64_t head_local = __atomic_load_8(&head, __ATOMIC_RELAXED);
    if (head_local > LIMIT)
      return LIMIT;
    else
      return head_local;
  }

  int get(uint16_t loc, uintptr_t& ret) {
    auto size = get_size();
    if (loc >= size)
      return -1;
    uintptr_t val;
    while (!(val = __atomic_load_8(&buf[loc], __ATOMIC_RELAXED)))
      ;

    ret = val;

    return 0;
  }
};

#ifdef STACK_CAPTURE

struct ThreadStackCap {
  bool non_recurse          = true;
  volatile bool initialized = false;
  volatile uint64_t top;
  volatile uint64_t bot;
};

thread_local ThreadStackCap cap;
static volatile uintptr_t mpsc_cap_buffer[1024] = {0};

struct StackCap {
  bool grows_down;
  uint64_t init_top;
  uint64_t init_bot;
  MPSCBuffer caps;

  uint64_t getStackPtr() {
    uint64_t a;
    return (uint64_t)&a;
  }

public:
  /* Old Constructors
  StackCap() : caps(1024)
  {
    uint64_t bot_val;

    uint64_t top_p = getStackPtr();
    grows_down = ((uint64_t)&bot_val > top_p);
    init_top = grows_down ? UINT64_MAX : 0;
    init_bot = grows_down ? 0 : UINT64_MAX;
  }

  StackCap(uint16_t lim, volatile uintptr_t* cap_buf) : caps(lim, cap_buf) {}
  */

  void setup() {
    uint64_t bot_val;

    uint64_t top_p = getStackPtr();
    grows_down     = ((uint64_t)&bot_val > top_p);
    init_top       = grows_down ? UINT64_MAX : 0;
    init_bot       = grows_down ? 0 : UINT64_MAX;
  }

  void capture_stack_info() {
    if (!cap.initialized) {
      this->setup();
      cap.top               = init_top;
      cap.bot               = init_bot;
      uintptr_t tl_cap_addr = (uintptr_t)&cap;
      caps.put(tl_cap_addr);
      __atomic_store_n(&cap.initialized, true, __ATOMIC_RELAXED);
    }

    uint64_t curr   = getStackPtr();
    bool change_top = grows_down ? (curr < cap.top) : (curr > cap.top);
    bool change_bot = grows_down ? (curr > cap.bot) : (curr < cap.bot);
    if (change_top)
      __atomic_store_8(&cap.top, curr, __ATOMIC_RELAXED);
    if (change_bot)
      __atomic_store_8(&cap.bot, curr, __ATOMIC_RELAXED);
  }

  /**
   * @return the maximum stack value
   * */
  uint64_t get_max() {
    uint64_t max = 0;
    for (uint16_t i = 0; i < caps.get_size(); i++) {
      uintptr_t tr_cap_addr = 0;
      caps.get(i, tr_cap_addr);
      ThreadStackCap* cont_cap = (ThreadStackCap*)tr_cap_addr;
      bool valid_vals =
          (cont_cap->top != init_top) && (cont_cap->bot != init_bot);
      uint64_t candidate_max = grows_down ? (cont_cap->bot - cont_cap->top)
                                          : (cont_cap->top - cont_cap->bot);
      if (valid_vals && (candidate_max > max))
        max = candidate_max;
    }

    return max;
  }

  /**
   * @param idx the index of the capacity you want to get
   * @param top a reference where the top of the stack should be put
   * @param bot a reference where the bottom of the stack should be put
   * @return tells you if the value at top and bot should be trusted (no errors
   *is 0)
   **/
  int get_top_bot(uint16_t idx, uint64_t& top, uint64_t& bot) {
    uintptr_t tr_cap_addr    = 0;
    int ret                  = caps.get(idx, tr_cap_addr);
    ThreadStackCap* cont_cap = (ThreadStackCap*)tr_cap_addr;
    if (ret != 0 || cont_cap == nullptr)
      return -1;
    top = cont_cap->top;
    bot = cont_cap->bot;
    return 0;
  }

  bool& is_non_recurse() { return *&cap.non_recurse; }

  /** This is a very dangerous function please be careful when calling this. */
  int reset() {
    for (uint16_t i = 0; i < caps.get_size(); i++) {
      uintptr_t tr_cap_addr = 0;
      caps.get(i, tr_cap_addr);
      ThreadStackCap* cont_cap = (ThreadStackCap*)tr_cap_addr;
      if (cont_cap == nullptr)
        return -1;
      cont_cap->top = init_top;
      cont_cap->bot = init_bot;
    }
    return 0;
  }
};

StackCap stack_capture = {
    .grows_down = true,
    .init_top   = 0,
    .init_bot   = 0,
    .caps       = {.head = 0, .buf = mpsc_cap_buffer, .LIMIT = 1024}};

#else
struct StackCap {
public:
  bool non_recurse = true;

  void capture_stack_info() {}

  uint64_t get_max() { return 0; }

  bool& is_non_recurse() { return non_recurse; }

  int reset() { return 0; }
};

StackCap stack_capture;

#endif

void cyg_profile_func_stack(void* this_fn, void* call_site) {
  (void)this_fn;
  (void)call_site;
  if (stack_capture.is_non_recurse()) {
    stack_capture.is_non_recurse() = false;
    stack_capture.capture_stack_info();
    stack_capture.is_non_recurse() = true;
  }
}

extern "C" {
void __cyg_profile_func_enter(void* this_fn, void* call_site) {
  cyg_profile_func_stack(this_fn, call_site);
}
void __cyg_profile_func_exit(void* this_fn, void* call_site) {
  cyg_profile_func_stack(this_fn, call_site);
}
}

#endif // GALOIS_RUNTIME_STACKTRACER_H
