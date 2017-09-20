#include "Galois/Galois.h"
#include "Galois/Timer.h"

#include <random>
#include <cstdio>
#include <time.h>

template<typename Gen>
void random_access(Gen& gen, int* buf, size_t size, size_t accesses) {
  std::uniform_int_distribution<size_t> randIndex(0, size - 1);
  for (unsigned i = 0; i < accesses; ++i) {
    size_t idx = randIndex(gen);
    buf[idx] += 1;
  }
}

struct run_local_helper {
  int* block;
  size_t seed;
  size_t size;
  run_local_helper(int* b, size_t s, size_t ss): block(b), seed(s), size(ss) { }
  run_local_helper() {}
  void operator()(unsigned int tid, unsigned int num) {
    std::mt19937 gen(seed + tid);
    std::uniform_int_distribution<int> randSeed;
    auto r = galois::block_range(block, block + size, tid, num);
    size_t d = std::distance(r.first, r.second);
    random_access(gen, r.first, d, d);
  }
};

void run_local(size_t seed, size_t mega) {
  size_t size = mega*1024*1024;
  int *block = (int*) malloc(size*sizeof(*block));

  // Assuming first touch policy
  galois::on_each(run_local_helper(block, seed, size));
  free(block);
}

struct run_interleaved_helper {
  int* block;
  size_t seed;
  size_t size;
  run_interleaved_helper(int* b, size_t s, size_t ss): block(b), seed(s), size(ss) { }
  run_interleaved_helper() {}
  void operator()(unsigned int tid, unsigned int num) {
    std::mt19937 gen(seed + tid);
    std::uniform_int_distribution<int> randSeed;
    auto r = galois::block_range(block, block + size, tid, num);
    size_t d = std::distance(r.first, r.second);
    random_access(gen, block, size, d);
  }
};

void run_interleaved(size_t seed, size_t mega, bool full) {
  size_t size = mega*1024*1024;
  auto ptr = galois::Substrate::largeMallocInterleaved(size * sizeof(int), full ? galois::Substrate::getThreadPool().getMaxThreads() : galois::runtime::activeThreads);
  int *block = (int*)ptr.get();
  galois::on_each(run_interleaved_helper(block, seed, size));
}

template<typename Fn>
long time_run(Fn fn) {
  galois::Timer t1;
  t1.start();
  fn();
  t1.stop();
  return t1.get();
}

struct F1 {
  size_t seed;
  size_t mega;
  F1(size_t s, size_t m): seed(s), mega(m) { }
  void operator()() { run_local(seed, mega); }
};

struct F2 {
  size_t seed;
  size_t mega;
  bool full;
  F2(size_t s, size_t m, bool f): seed(s), mega(m), full(f) { }
  void operator()() { run_interleaved(seed, mega, full); }
};

int main(int argc, char** argv) {
  unsigned M = galois::Substrate::getThreadPool().getMaxThreads() / 2;
  size_t mega = 1;
  if (argc > 1)
    mega = atoi(argv[1]);
  if (!mega)
    mega = 200;

  size_t seed = time(NULL);
  printf("Working set: %zu MB\n\n", mega);
  printf("Effective random-access bandwidth (MB/s)\n");
  printf("T    LOCAL    INTERLEAVE    FULL-INTERLEAVE\n");
  for (unsigned threads = 1; threads <= M; ++threads) {
    galois::setActiveThreads(threads);

    long local_millis = time_run(F1(seed, mega));
    long interleave_millis = time_run(F2(seed, mega, false));
    long full_interleave_millis = time_run(F2(seed, mega, true));
    double mb = mega / (double) sizeof(int);
    // 4 + length of column header
    printf("%4d %8.2f %13.2f %18.2f\n", 
        threads,
        mb / local_millis * 1000.0,
        mb / interleave_millis * 1000.0,
        mb / full_interleave_millis * 1000.0);
  }
}
