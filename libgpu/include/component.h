/*
   component.h

   Implements ComponentSpace. Part of the GGC source code.
   Originally derived from the LonestarGPU 2.0 source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   TODO: relicense
*/

struct ComponentSpace {
  ComponentSpace(unsigned nelements);

  __device__ unsigned numberOfElements();
  __device__ unsigned numberOfComponents();
  __device__ bool isBoss(unsigned element);
  __device__ unsigned find(unsigned lelement, bool compresspath = true);
  __device__ bool unify(unsigned one, unsigned two);
  __device__ void print1x1();
  __host__ void print();
  __host__ void copy(ComponentSpace& two);
  void dump_to_file(const char* F);
  void allocate();
  void init();
  unsigned numberOfComponentsHost();

  unsigned nelements;
  unsigned *ncomponents, // number of components.
      *complen,          // lengths of components.
      *ele2comp;         // components of elements.
};
ComponentSpace::ComponentSpace(unsigned nelements) {
  this->nelements = nelements;

  allocate();
  init();
}

void ComponentSpace::dump_to_file(const char* F) {
  static FILE* f;
  static unsigned* mem;

  if (!f) {
    f   = fopen(F, "w");
    mem = (unsigned*)calloc(nelements, sizeof(unsigned));
  }

  assert(cudaMemcpy(mem, ele2comp, nelements * sizeof(unsigned),
                    cudaMemcpyDeviceToHost) == cudaSuccess);

  int i;
  for (i = 0; i < nelements; i++) {
    int boss = i;
    do {
      boss = mem[boss];
    } while (boss != mem[boss]);
    fprintf(f, "%d %d %d\n", i, mem[i], boss);
  }

  fprintf(f, "\n");
}

void ComponentSpace::copy(ComponentSpace& two) {
  assert(cudaMemcpy(two.ncomponents, ncomponents, sizeof(unsigned),
                    cudaMemcpyDeviceToDevice) == 0);
  assert(cudaMemcpy(two.ele2comp, ele2comp, sizeof(unsigned) * nelements,
                    cudaMemcpyDeviceToDevice) == 0);
  assert(cudaMemcpy(two.complen, complen, sizeof(unsigned) * nelements,
                    cudaMemcpyDeviceToDevice) == 0);
}
__device__ void ComponentSpace::print1x1() {
  printf("\t\t-----------------\n");
  for (unsigned ii = 0; ii < nelements; ++ii) {
    printf("\t\t%d -> %d\n", ii, ele2comp[ii]);
  }
  printf("\t\t-----------------\n");
}
__global__ void print1x1(ComponentSpace cs) { cs.print1x1(); }
__host__ void ComponentSpace::print() { ::print1x1<<<1, 1>>>(*this); }
__device__ unsigned ComponentSpace::numberOfElements() { return nelements; }
__device__ unsigned ComponentSpace::numberOfComponents() {
  return *ncomponents;
}
unsigned ComponentSpace::numberOfComponentsHost() {
  unsigned hncomponents = 0;
  check_cuda(cudaMemcpy(&hncomponents, ncomponents, sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
  return hncomponents;
}
void ComponentSpace::allocate() {
  check_cuda(cudaMalloc((void**)&ncomponents, 1 * sizeof(unsigned)));
  check_cuda(cudaMalloc((void**)&complen, nelements * sizeof(unsigned)));
  check_cuda(cudaMalloc((void**)&ele2comp, nelements * sizeof(unsigned)));
}
__global__ void dinitcs(unsigned nelements, unsigned* complen,
                        unsigned* ele2comp) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < nelements) {
    // elements[id] 	= id;
    complen[id]  = 1;
    ele2comp[id] = id;
  }
}
void ComponentSpace::init() {
  // init the elements.
  unsigned blocksize = 256; ////
  unsigned nblocks   = (nelements + blocksize - 1) / blocksize;
  dinitcs<<<nblocks, blocksize>>>(nelements, complen, ele2comp);
  // init number of components.
  check_cuda(cudaMemcpy(ncomponents, &nelements, sizeof(unsigned),
                        cudaMemcpyHostToDevice));
}
__device__ bool ComponentSpace::isBoss(unsigned element) {
  return atomicCAS(&ele2comp[element], element, element) == element;
}
__device__ unsigned ComponentSpace::find(unsigned lelement,
                                         bool compresspath /*= true*/) {
  // do we need to worry about concurrency in this function?
  // for other finds, no synchronization necessary as the data-structure is a
  // tree. for other unifys, synchornization is not required considering that
  // unify is going to affect only bosses, while find is going to affect only
  // non-bosses.
  unsigned element = lelement;
  while (isBoss(element) == false) {
    element = ele2comp[element];
  }
  if (compresspath)
    ele2comp[lelement] = element; // path compression.
  return element;
}
__device__ bool ComponentSpace::unify(unsigned one, unsigned two) {
  // if the client makes sure that one component is going to get unified as a
  // source with another destination only once, then synchronization is
  // unnecessary. while this is true for MST, due to load-balancing in if-block
  // below, a node may be source multiple times. if a component is source in one
  // thread and destination is another, then it is okay for MST.
  do {
    if (!isBoss(one))
      return false;
    if (!isBoss(two))
      return false;

    unsigned onecomp = one;
    unsigned twocomp = two;
    // unsigned onecomp = find(one, false);
    // unsigned twocomp = find(two, false);

    if (onecomp == twocomp)
      return false; // "duplicate" edges due to symmetry

    unsigned boss        = twocomp;
    unsigned subordinate = onecomp;
    // if (complen[onecomp] > complen[twocomp]) {	// one is larger, make it the
    // representative: can create cycles.
    if (boss < subordinate) { // break cycles by id.
      boss        = onecomp;
      subordinate = twocomp;
    }
    // merge subordinate into the boss.
    // ele2comp[subordinate] = boss;

    unsigned oldboss = atomicCAS(&ele2comp[subordinate], subordinate, boss);
    if (oldboss != subordinate) { // someone else updated the boss.
      // we need not restore the ele2comp[subordinate], as union-find ensures
      // correctness and complen of subordinate doesn't matter.
      one = oldboss;
      two = boss;
      return false;
    } else {
      dprintf("\t\tunifying %d -> %d (%d)\n", subordinate, boss);
      atomicAdd(&complen[boss], complen[subordinate]);
      // complen[boss] += complen[subordinate];
      // complen[subordinate] doesn't matter now, since find() will find its
      // boss.

      // a component has reduced.
      unsigned ncomp = atomicSub(ncomponents, 1);
      // atomicDec(ncomponents, nelements);
      dprintf("\t%d: ncomponents = %d\n", threadIdx.x, ncomp);
      return true;
    }
  } while (true);
}
