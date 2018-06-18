/*
 * unexpected_eof.h
 *
 *  Created on: Aug 25, 2014
 *      Author: marcos
 */

#ifndef UNEXPECTED_EOF_H_
#define UNEXPECTED_EOF_H_

#include <exception>
#include <sstream>
#include <string>

class unexpected_eof : public std::exception {
  unsigned l, c;
  std::string msg;

public:
  unexpected_eof(unsigned l, unsigned c);
  unexpected_eof(unsigned l, unsigned c, std::string msg);
  virtual const char* what() const throw();
  virtual ~unexpected_eof() throw();
};

#endif /* UNEXPECTED_EOF_H_ */
