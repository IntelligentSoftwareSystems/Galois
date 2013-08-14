#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <vector>
#include <list>
#include "Point2D/Tier.hxx"
#include "EquationSystem.h"

class Preprocessor {
	virtual template<typename Input>
	std::vector<EquationSystem *>* preprocess(std::list<Input*>* input);
};

class Mes2DPreprocessor : public Preprocessor<Tier> {
public:
	std::vector<EquationSystem*>* preprocess(std::list<Tier *> *tier_list);
};

#endif /* PREPROCESSOR_H_ */
