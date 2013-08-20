#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <vector>
#include <list>
#include "Point3D/Tier.hxx"
#include "Point2D/Tier.hxx"
#include "EquationSystem.h"

class Preprocessor {
	template<typename Input>
	std::vector<EquationSystem *>* preprocess(std::list<Input*>* input);
};

class Mes2DPreprocessor : public Preprocessor {
public:
	std::vector<EquationSystem*>* preprocess(std::list<D2::Tier *> *tier_list);
};

class Mes3DPreprocessor : public Preprocessor {
public:
	std::vector<EquationSystem*>* preprocess(std::list<D3::Tier *> *tier_list);
};

#endif /* PREPROCESSOR_H_ */
