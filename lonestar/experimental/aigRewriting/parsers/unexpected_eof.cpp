/*
 * unexpected_eof.cpp
 *
 *  Created on: Aug 25, 2014
 *      Author: marcos
 */

#include "unexpected_eof.h"

unexpected_eof::unexpected_eof(unsigned l, unsigned c) : std::exception() {
	this->l = l;
	this->c = c;
	this->msg = "";
}

unexpected_eof::unexpected_eof(unsigned l, unsigned c, std::string msg) : std::exception() {
	this->l = l;
	this->c = c;
	this->msg = msg;
}

const char* unexpected_eof::what() const throw () {
	std::stringstream ret;
	ret << "Unexpected eof in line-" << l << " char-" << c;
	if(msg == "")ret << std::endl;
	else ret << ". Last token: " << msg << std::endl;
	return ret.str().c_str();
}

unexpected_eof::~unexpected_eof() throw () {
}
