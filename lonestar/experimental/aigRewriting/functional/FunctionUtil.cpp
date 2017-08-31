/*
 * FunctionUtil.cpp
 *
 *  Created on: 14/02/2017
 *      Author: possani
 */

#include "../functional/FunctionUtil.h"

#include "../functional/FunctionHandler.h"

namespace Functional {

FunctionUtil::FunctionUtil( StringFunctionMap & entries, BitVectorPool & functionPool, int nVars, int nWords ) : literals( entries ), functionPool( functionPool ) {
	this->nVars = nVars;
	this->nWords = nWords;
	this->currentToken = EMPTY;
}

FunctionUtil::~FunctionUtil() {
}

word * FunctionUtil::parseExpression( std::string expression ) {

	std::istringstream functionality( expression );
	return expr1( functionality );
}

word * FunctionUtil::expr1( std::istringstream & expression ) {

	word * lhs = expr2( expression );

	for ( ; ; ) {
		switch ( this->currentToken ) {
			case ORop: {
				word * result = this->functionPool.getMemory();
				word * rhs = expr2( expression );
				Functional::OR( result, lhs, rhs, this->nWords );
				lhs = result;
				break;
			}

			default:
				return lhs;
		}
	}
}

word * FunctionUtil::expr2( std::istringstream & expression ) {

	word * lhs = term( expression );

	for ( ; ; ) {
		switch ( this->currentToken ) {
			case XORop: {
				word * result = this->functionPool.getMemory();
				word * rhs = term( expression );
				Functional::XOR( result, lhs, rhs, this->nWords );
				lhs = result;
				break;
			}

			default:
				return lhs;
		}
	}
}

word * FunctionUtil::term( std::istringstream & expression ) {

	word * lhs = prim( expression );

	for ( ; ; ) {
		switch ( this->currentToken ) {
			case ANDop: {
				word * result = this->functionPool.getMemory();
				word * rhs = prim( expression );
				Functional::AND( result, lhs, rhs, this->nWords );
				lhs = result;
				break;
			}

			default:
				return lhs;
		}
	}
}

word * FunctionUtil::prim( std::istringstream & expression ) {

	getToken( expression );

	switch ( this->currentToken ) {

		case LIT: {
			getToken( expression );
			StringFunctionMap::iterator it = literals.find( this->tokenValue );
			if( it != literals.end() ) {
				word * var = it->second.first;
				word * literal = this->functionPool.getMemory();
				Functional::copy( literal, var, this->nWords );
				return literal;
			}
			else {
				std::cout << "ERROR: Literal ( " << tokenValue << " ) not found!" << std::endl;
				exit(1);
			}
		}

		case NOTop: {
			word * function = prim( expression );
			Functional::NOT( function, function, this->nWords );
			return function;
		}

		case LP: {
			word * function = expr1( expression );
			if ( currentToken != RP ) {
				std::cout << "ERROR: current token = " << currentToken << std::endl;
				exit(1);
			}
			getToken( expression ); //eat )
			return function;
		}

		default:
			break;
	}
}

Token FunctionUtil::getToken( std::istringstream & expression ) {

	char ch = 0;
	expression >> ch;

	switch ( ch ) {
		case 0: {
			return this->currentToken = END;
		}

		case ';':
		case '*':
		case '+':
		case '^':
		case '!':
		case '(':
		case ')':
		case '=':
			return this->currentToken = Token( ch );

		default: {
			if ( isalpha( ch ) ) {
				this->tokenValue = "";
				for ( ; isalnum( ch ) && !expression.eof(); expression >> ch ) {
					tokenValue += ch;  //; needed at the end of string
				}
				expression.putback( ch );
				return this->currentToken = LIT;
			}
			return this->currentToken = END;
		}
	}
}

word * FunctionUtil::parseHexa( std::string hexa ) {

	if ( (hexa.at( 0 ) == '0') && (hexa.at( 1 ) == 'x') ) {
		hexa = hexa.substr( 2 );
	}

	word * function = this->functionPool.getMemory();
	unsigned long int value;
	std::stringstream ss;

	if ( this->nVars < 6 ) {
		ss << std::hex << hexa;
		ss >> value;
		function[0] = static_cast<unsigned long int>( value );
		return function;
	}
	else {
		int lhs = hexa.size() - 16;
		int i = 0;
		while ( lhs >= 0 ) {
			std::string currentHexa = hexa.substr( lhs, 16 );
			ss.clear();
			ss.str("");
			ss << std::hex << currentHexa;
			ss >> value;
			function[ i++ ] = static_cast<unsigned long int>( value );
			lhs -= 16;
		}
		return function;
	}
}

}
