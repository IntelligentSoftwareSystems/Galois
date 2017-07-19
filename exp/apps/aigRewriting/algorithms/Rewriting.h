#ifndef REWRITING_H_
#define REWRITING_H_

#include "Aig.h"
#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Timer.h"

namespace algorithm {

class Rewriting {

private:

	aig::Aig & aig;

public:

	Rewriting( aig::Aig & aig );

	virtual ~Rewriting();

	void run( int nOutputs, int nInputs, int nLevels, int nFanout, int cutSizeLimit, int verbose );
};

} /* namespace algorithm */

namespace alg = algorithm;

#endif /* REWRITING_H_ */
