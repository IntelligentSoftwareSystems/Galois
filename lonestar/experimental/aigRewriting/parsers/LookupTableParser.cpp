#include "LookupTableParser.h"

#include <fstream>
#include <sstream>

namespace lookuptables {

LookupTableParser::LookupTableParser() {

}

LookupTableParser::~LookupTableParser() {

}

void LookupTableParser::parseFile( std::string fileName, LookupTableElement ** lookupTable ) {

	std::ifstream file( fileName ); 
	std::string line;
	std::string token, exp;
	char lit, lev;

	int i = 0;
	int j = 0;

	while ( std::getline( file, line ) ) {

		if ( line.at(0) == '#' ) {
			i++;
			j = 0;
			continue;
		}	
	
		std::stringstream tokenizer;
		tokenizer << line;
		
		std::getline( tokenizer, exp, ';' );

		std::getline( tokenizer, token, ';' );
		lit = std::stoi( token );
		
		std::getline( tokenizer, token, ';' );
		lev = std::stoi( token );
		
		lookupTable[i][j] = LookupTableElement( exp, lit, lev );	

		j++;
	}
}

} /* namespace lookuptables */
