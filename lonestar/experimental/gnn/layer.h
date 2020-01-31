#pragma once

#include <queue>
#include <vector>
#include <limits>
#include <memory>
#include <string>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include "types.h"
#include "math_functions.hpp"
/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 * - in_shape            ... specify input data shapes
 * - out_shape           ... specify output data shapes
 * - layer_type          ... name of layer
 **/

class layer {
public:
	layer() { }
	virtual ~layer() = default;
	virtual void forward(const std::vector<FV> &in_data, std::vector<FV> &out_data) = 0;
	virtual void backward(const std::vector<FV> &in_data, const std::vector<FV> &out_data,
			std::vector<FV> &out_grad, std::vector<FV> &in_grad) = 0;
	virtual std::string layer_type() const = 0;
	virtual void set_param(Graph *g, FV2D *w, FV2D *q, FV *d, LabelList *lab) = 0;
protected:
	std::string name;
};
