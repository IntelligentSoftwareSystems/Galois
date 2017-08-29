/*
 * semantic_error.h
 *
 *  Created on: Aug 25, 2014
 *      Author: marcos
 */

#ifndef SEMANTIC_ERROR_H_
#define SEMANTIC_ERROR_H_

#include <exception>
#include <sstream>
#include <string>

class semantic_error: public std::exception {
	unsigned l,c;
	std::string msg;
public:
	semantic_error(unsigned l, unsigned c, std::string msg = "");
	virtual const char * what() const throw ();
	virtual ~semantic_error() throw ();
};

#endif /* SEMANTIC_ERROR_H_ */
