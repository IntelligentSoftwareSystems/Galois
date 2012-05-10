#ifndef A_STLSORT_INCLUDED
#define A_STLSORT_INCLUDED
#include <algorithm>

template <class E, class BinPred>
void compSort(E* A, int n, BinPred f) { std::sort(A,A+n,f);}

#endif // _A_STLSORT_INCLUDED
