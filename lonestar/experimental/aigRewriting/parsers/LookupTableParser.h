#ifndef LOOKUPTABLEPARSER_H_
#define LOOKUPTABLEPARSER_H_

#include <iostream>

namespace lookuptables {

typedef struct lookupTableElement {
	std::string expression;
	char literals;
	char levels;

	lookupTableElement() {
		expression = "";
		literals = 0;
		levels = 0;
	}
	
	lookupTableElement( std::string & expression, char literals, char levels ) : expression( expression ), literals( literals ), levels( levels ) { }

} LookupTableElement;


class LookupTableParser {

private:
	


public:

	LookupTableParser();
	
	~LookupTableParser();
	
	void parseFile( std::string fileName, LookupTableElement ** lookupTable );


};

} /* namespace lookuptables */

#endif /* LOOKUPTABLEPARSER_H_ */ 
