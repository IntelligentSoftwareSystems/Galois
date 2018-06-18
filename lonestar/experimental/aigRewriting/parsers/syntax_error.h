/*
 * syntax_error.h
 *
 *  Created on: Aug 25, 2014
 *      Author: marcos
 */

#ifndef SYNTAXEXEPTION_H_
#define SYNTAXEXEPTION_H_

#include <exception>
#include <sstream>
#include <string>

class syntax_error : public std::exception {
  unsigned l, c;
  std::string msg;

public:
  syntax_error(unsigned l, unsigned c, std::string msg = "");
  virtual const char* what() const throw();
  virtual ~syntax_error() throw();
};

#endif /* SYNTAXEXEPTION_H_ */
