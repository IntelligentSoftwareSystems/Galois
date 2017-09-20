#ifndef AIGWRITER_H_
#define AIGWRITER_H_

#include <fstream>
#include <iostream>
#include <string>

#include "../subjectgraph/aig/Aig.h"
#include "galois/Galois.h"

typedef aig::Aig Aig;

class AigWriter {

private:

	std::ofstream aigerFile;
	std::string path;

	void writeAagHeader( Aig & aig );
	void writeLatchesAag( Aig & aig );
	void writeAndsAag( Aig & aig );
	
	void writeAigHeader( Aig & aig );
	void writeLatchesAig( Aig & aig );
	void writeAndsAig( Aig & aig );

	void writeInputs( Aig & aig );
	void writeOutputs( Aig & aig );
	void writeSymbolTable( Aig & aig );

	void encode( unsigned x );

public:

	AigWriter();
	AigWriter( std::string path );
	virtual ~AigWriter();

	void setFile( std::string path );
	bool isOpen();
	void close();

	void writeAag( Aig & aig );
	void writeAig( Aig & aig );
};

#endif /* AIGWRITER_H_ */
