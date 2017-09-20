#ifndef WINDOWING_WINDOWING_H_
#define WINDOWING_WINDOWING_H_

#include "Aig.h"
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/Timer.h"

namespace algorithm {

class Windowing {

private:

	aig::Aig & aig;

public:

	Windowing( aig::Aig & aig );
	virtual ~Windowing();

	void runStandardWindowing( int nFanins, int nFanouts );
	void runReconvDrivenWindowing( int nInputs, int nOutputs, int nLevels );
};

} /* namespace algorithm */

namespace alg = algorithm;

#endif /* WINDOWING_H_ */
