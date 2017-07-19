/*
 * SyntaxExeption.cpp
 *
 *  Created on: Aug 25, 2014
 *      Author: marcos
 */

#include "syntax_error.h"

syntax_error::syntax_error(unsigned l, unsigned c, std::string msg) : std::exception(){
	this->l = l;
	this->c = c;
	this->msg = msg;
}

const char * syntax_error::what() const throw (){
	std::stringstream ret;
	ret << "Syntax error in line-" << l << " char-" << c << ": " << msg << std::endl;
	return ret.str().c_str();
}

syntax_error::~syntax_error() throw () {
}
