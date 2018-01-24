/*
 
 @Vinicius Possani 
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#ifndef RECONVDRIVENCUT_H_
#define RECONVDRIVENCUT_H_

#include "Aig.h"

#include <unordered_set>

namespace algorithm {

typedef struct RDCutData_ {

	std::unordered_set< aig::GNode > visited;
	std::unordered_set< aig::GNode > leaves;

} RDCutData;

typedef galois::substrate::PerThreadStorage< RDCutData > PerThreadRDCutData;

class ReconvDrivenCut {

private:

	aig::Aig & aig;
	PerThreadRDCutData perThreadRDCutData;

public:

	ReconvDrivenCut( aig::Aig & aig );

	virtual ~ReconvDrivenCut();

	void run( int cutSizeLimit );
};

} /* namespace algorithm */

namespace alg = algorithm;

#endif /* RECONVDRIVENCUT_H_ */
