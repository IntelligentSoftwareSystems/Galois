#ifndef AIGPARSER_H_
#define AIGPARSER_H_
#include <string>
#include <fstream>
#include <sstream>

#include "semantic_error.h"
#include "syntax_error.h"
#include "unexpected_eof.h"
#include "../misc/util/utilString.h"
#include "../subjectgraph/aig/Aig.h"

class AigParser {

private:
	unsigned currLine;
	unsigned currChar;
	std::ifstream file;
	int m, i, l, o, a;
	std::vector<int> inputs, outputs;
	std::vector< std::tuple< int, int, bool > > latches;
	std::vector< std::tuple< int, int, int > > ands;
	std::vector<std::string> inputNames, latchNames, outputNames;

	std::vector< aig::GNode > nodes;
	std::string designName;

	unsigned char parseChar();
	char parseByte();
	int parseInt( std::string delimChar );
	std::string parseString( std::string delimChar );
	bool parseBool( std::string delimChar );
	unsigned decode();

	void resize();

	void parseAagHeader();
	void parseAigHeader();
	void parseAagInputs();
	void parseAigInputs();
	void parseAagLatches();
	void parseAigLatches();
	void parseOutputs();
	void parseAagAnds();
	void parseAigAnds();
	void parseSymbolTable();

	void createAig( aig::Aig & aig );
	void createConstant( aig::Aig & aig );
	void createInputs( aig::Aig & aig );
	void createLatches( aig::Aig & aig );
	void createOutputs( aig::Aig & aig );
	void createAnds( aig::Aig & aig );

	void connectLatches( aig::Aig & aig );
	void connectOutputs( aig::Aig & aig );
	void connectAnds( aig::Aig & aig );

public:
	AigParser();
	AigParser(std::string fileName);
	virtual ~AigParser();
	void open(std::string fileName);
	bool isOpen() const;
	void close();
	void parseAag( aig::Aig & aig );
	void parseAig( aig::Aig & aig );

	int getI();
	int getL();
	int getO();
	int getA();
	int getE( aig::Aig & aig );
};

#endif /* AIGPARSER_H_ */
