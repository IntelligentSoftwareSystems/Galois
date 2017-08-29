/*
 * semantic_error.cpp
 *
 *  Created on: Aug 25, 2014
 *      Author: marcos
 */

#include "semantic_error.h"


semantic_error::semantic_error(unsigned l, unsigned c, std::string msg) : exception() {
	this->l = l;
	this->c = c;
	this->msg = msg;
}

const char* semantic_error::what() const throw () {
	std::stringstream ret;
	ret << "Semantic error in line-" << l << " char-" << c << ": " << msg << std::endl;
	return ret.str().c_str();
}

semantic_error::~semantic_error() throw () {
}

