#ifndef A_STLSORT_INCLUDED
#define A_STLSORT_INCLUDED
#include <parallel/algorithm>

template <class E, class BinPred>
void compSort(E* A, int n, BinPred f) { __gnu_parallel::sort(A,A+n,f);}

#endif // _A_STLSORT_INCLUDED
