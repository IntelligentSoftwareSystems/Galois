#include <numa.h>

int main() {
  struct bitmask* nm = numa_allocate_nodemask();
  numa_bitmask_setbit(nm, 1);
  numa_alloc_interleaved_subset(1, nm);
  numa_free_nodemask(nm);
  return 0;
}
