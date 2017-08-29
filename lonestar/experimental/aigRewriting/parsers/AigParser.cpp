#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include <AigParser.h>
#include <iostream>
#include <algorithm>

AigParser::AigParser() {
	currLine = 0;
	currChar = 0;
}

AigParser::AigParser(std::string fileName) {
	currLine = 1;
	currChar = 0;
	open(fileName);
}

void AigParser::open(std::string fileName) {
	close();
	currLine = 1;
	currChar = 0;
	file.open(fileName.c_str());
}

bool AigParser::isOpen() const {
	return file.is_open();
}

void AigParser::close() {
	if (isOpen()) {
		file.close();
		currLine = 0;
		currChar = 0;
	}
}

int AigParser::parseInt( std::string delimChar ) {
	unsigned char c;
	int result;
	bool done = false;
	std::stringstream buffer;
	c = parseChar();
	if (!isdigit(c))
		throw syntax_error(currLine, currChar, format("Expected integer but found: ASCII %d", c));
	buffer << c;
	while (!done) {
		c = parseChar();
		if (isdigit(c)) {
			buffer << c;
		} else if (delimChar.find(c) != std::string::npos) {
			buffer >> result;
			done = true;
		} else {
			throw syntax_error(currLine, currChar, format("Expected integer but found: ASCII %d", c));
		}
	}
	return result;
}

std::string AigParser::parseString(std::string delimChar) {
	bool done = false;
	unsigned char c;
	std::stringstream buffer;
	c = parseChar();
	if (delimChar.find(c) != std::string::npos || c == '\0') {
		throw syntax_error(currLine, currChar, format("Expected integer but found: ASCII %d", c));
	}
	buffer << c;
	while (!done) {
		c = parseChar();
		if (delimChar.find(c) != std::string::npos || c == '\0') {
			done = true;
		} else {
			buffer << c;
		}
	}
	return buffer.str();
}

void AigParser::parseAagHeader() {
	std::string aag = parseString(" ");
	if ( aag.compare( "aag" ) != 0 ) {
		throw syntax_error(1, 0, "Expected aag header");
	}
	m = parseInt(" ");
	i = parseInt(" ");
	l = parseInt(" ");
	o = parseInt(" ");
	a = parseInt("\n");

	if (m != (i + l + a)) {
		throw semantic_error(1, 4, "Incorrect value for M");
	}
	resize();
}

void AigParser::parseAigHeader() {
	std::string aig = parseString(" ");
	if ( aig.compare( "aig" ) != 0 ) {
		throw syntax_error(1, 0, "Expected aig header");
	}
	m = parseInt(" ");
	i = parseInt(" ");
	l = parseInt(" ");
	o = parseInt(" ");
	a = parseInt("\n");
	
	if (m != (i + l + a)) {
		throw semantic_error(1, 4, "Incorrect value for M");
	}
	resize();
}

unsigned char AigParser::parseChar() {
	int result = file.get();
	if (file.eof()) {
		throw unexpected_eof(currLine, currChar);
	}
	if (result == '\r') {
		result = file.get();
		if (file.eof()) {
			throw unexpected_eof(currLine, currChar);
		}
	}
	if (result == '\n') {
		currLine++;
		currChar = 0;
	} else {
		currChar++;
	}
	return result;
}

void AigParser::parseAagInputs() {
	for (int in = 0; in < i; in++) {
		inputs[in] = parseInt("\n");
	}
}

void AigParser::parseAigInputs() {
	int x = 2;
	for (int in = 0; in < i; in++) {
		inputs[in] = x;
		x = x + 2;
	}
}

void AigParser::parseOutputs() {
	for (int in = 0; in < o; in++) {
		outputs[in] = parseInt("\n");
	}
}

void AigParser::parseAagAnds() {
	for (int in = 0; in < a; in++) {
		int lhs = parseInt(" ");
		int rhs0 = parseInt(" ");
		int rhs1 = parseInt("\n");
		ands[in] = std::make_tuple( lhs, rhs0, rhs1 );
	}
}

void AigParser::parseAagLatches() {
	unsigned line;
	for (int in = 0; in < l; in++) {
		int lhs = parseInt(" ");
		line = currLine;
		int rhs = parseInt("\n");
		bool init;
		if (line == currLine) {
			init = parseBool("\n");
		} else {
			init = false;
		}
		latches[in] = std::make_tuple( lhs, rhs, init );
	}
}

void AigParser::parseAigLatches() {
	unsigned line;
	int lhs = i * 2 + 2;
	for (int in = 0; in < l; in++) {
		line = currLine;
		int rhs = parseInt("\n");
		bool init;
		if (line == currLine) {
			init = parseBool("\n");
		} else {
			init = false;
		}
		latches[in] = std::make_tuple( lhs, rhs, init );
		lhs = lhs + 2;
	}
}

bool AigParser::parseBool(std::string delimChar) {
	bool result;
	int c = parseChar();
	switch (c) {
	case '0':
		result = false;
		break;
	case '1':
		result = true;
		break;
	default:
		throw syntax_error(currLine, currChar, format("Expected Boolean (0 or 1) but found: ASCII %d", c));
	}
	c = parseChar();
	if (delimChar.find(c) == std::string::npos) {
		throw syntax_error(currLine, currChar, format("Expected Boolean (0 or 1) but found: ASCII %d", c));
	}
	return result;
}

void AigParser::parseSymbolTable() {
	int c, n;
	while (true) {
		try {
			c = parseChar();
		} catch (unexpected_eof& e) {
			return;
		}
		switch (c) {
		case 'i':
			n = parseInt(" ");
			if (n >= i)
				throw semantic_error(currLine, currChar, "Input number greater than number of inputs");
			inputNames[n] = parseString("\n");
			break;
		case 'l':
			n = parseInt(" ");
			if (n >= l)
				throw semantic_error(currLine, currChar, "Latch number greater than number of latches");
			latchNames[n] = parseString("\n");
			break;
		case 'o':
			n = parseInt(" ");
			if (n >= o)
				throw semantic_error(currLine, currChar, "Output number greater than number of outputs");
			outputNames[n] = parseString("\n");
			break;
		case 'c':
			c = parseChar();
			if (c != '\n' && c != '\r')
				throw syntax_error(currLine, currChar);
			if (file.peek() != 0)
				designName = parseString("\n\0");
			else
				designName = "Unnamed";
			return;
		}
	}
}

unsigned AigParser::decode() {
	unsigned x = 0, in = 0;
	unsigned char c;
	while ((c = parseByte()) & 0x80)
		x |= (c & 0x7f) << (7 * in++);
	return x | (c << (7 * in));
}

void AigParser::parseAigAnds() {
	int delta0, delta1;
	int lhs = (i + l) * 2 + 2;
	for (int in = 0; in < a; in++) {
		delta0 = decode();
		delta1 = decode();
		int rhs0 = lhs - delta0;
		if ( rhs0 < 0 ) {
			throw semantic_error(currLine, currChar, format("Negative rhs0: %d", rhs0));
		}
		int rhs1 = rhs0 - delta1;
		if ( rhs1 < 0 ) {
			throw semantic_error(currLine, currChar, format("Negative rhs0: %d", rhs1));
		}
		ands[in] = std::make_tuple( lhs, rhs0, rhs1 );
		lhs = lhs + 2;
	}
}

char AigParser::parseByte() {
	char byte;
	file.read(&byte, 1);
	if (file.eof()) {
		throw unexpected_eof(currLine, currChar);
	}
	if (byte == '\n') {
		currLine++;
		currChar = 0;
	} else {
		currChar++;
	}
	return byte;
}

void AigParser::parseAag( aig::Aig & aig ) {
	parseAagHeader();
	parseAagInputs();
	parseAagLatches();
	parseOutputs();
	parseAagAnds();
	parseSymbolTable();
	createAig( aig );
}

void AigParser::parseAig( aig::Aig & aig ) {	
	parseAigHeader();
	parseAigInputs();
	parseAigLatches();
	parseOutputs();
	parseAigAnds();
	parseSymbolTable();
	createAig( aig );
}

void AigParser::resize() {
	inputs.resize(i);
	inputNames.resize(i);
	latches.resize(l);
	latchNames.resize(l);
	outputs.resize(o);
	outputNames.resize(o);
	ands.resize(a);
	nodes.resize(m + o + 1);
}

void AigParser::createAig( aig::Aig & aig ) {
	aig.setDesignName( this->designName );
	createConstant( aig );
	createInputs( aig );
	createLatches( aig );	
	createOutputs( aig );
	createAnds( aig );
	connectAnds( aig );
	connectLatches( aig );
	connectOutputs( aig );
}

void AigParser::createConstant( aig::Aig & aig ) {
	// Node Data
	aig::NodeData nodeData;
	nodeData.id = 0;
	// nodeData.name = "Const_0";
	nodeData.counter = 0;
	nodeData.type = aig::NodeType::CONST;
	// AIG Node
	aig::Graph & graph = aig.getGraph();
	aig::GNode constNode;
	constNode = graph.createNode( nodeData );
	graph.addNode( constNode );
	nodes[ nodeData.id ] = constNode;
}

void AigParser::createInputs( aig::Aig & aig ) {
	aig::Graph & graph = aig.getGraph();
	for (int in = 0; in < i; in++) {
		// Node Data
		aig::NodeData nodeData;
		nodeData.id = inputs[in] / 2;
		// if (inputNames[in].empty()) {
		// 	std::stringstream sName;
		// 	sName << "i" << in;
		// 	nodeData.name = sName.str();
		// }
		// else {
		// 	nodeData.name = inputNames[in];
		// }
		nodeData.counter = 0;
		nodeData.type = aig::NodeType::PI;
		// AIG Node
 		aig::GNode inputNode;
 		inputNode = graph.createNode( nodeData );
 		graph.addNode( inputNode );
		aig.getInputNodes().push_back( inputNode );
		nodes[ nodeData.id ] = inputNode;
	}

	aig.setInputNames( this->inputNames );
}

void AigParser::createLatches( aig::Aig & aig ) {
	aig::Graph & graph = aig.getGraph();
	for (int in = 0; in < l; in++) {
		// Node Data
		aig::NodeData nodeData;
		nodeData.id = ( std::get<0>( latches[in] ) / 2 );
		// if (latchNames[in].empty()) {
		// 	std::stringstream sName;
		// 	sName << "l" << in;
		// 	nodeData.name = sName.str();
		// }
		// else {
		// 	nodeData.name = latchNames[in];
		// }
		nodeData.counter = 0;
		nodeData.type = aig::NodeType::LATCH;
		// NodeData.initialValue = std::get<3>( latches[in] ); // FIXME
		// AIG Node
 		aig::GNode latchNode;
 		latchNode = graph.createNode( nodeData );
 		graph.addNode( latchNode );
		aig.getLatchNodes().push_back( latchNode );
		nodes[ nodeData.id ] = latchNode;
	}

	aig.setLatchNames( this->latchNames );
}

void AigParser::createOutputs( aig::Aig & aig ) {
	aig::Graph & graph = aig.getGraph();
	for (int in = 0; in < o; in++) {
		// Node Data
		aig::NodeData nodeData;
		nodeData.id = m + in + 1;
		// if (outputNames[in].empty()) {
		// 	std::stringstream sName;
		// 	sName << "o" << in;
		// 	nodeData.name = sName.str();
		// }
		// else {
		// 	nodeData.name = outputNames[in];
		// }
		nodeData.counter = 0;
		nodeData.type = aig::NodeType::PO;
		// AIG Node
 		aig::GNode outputNode;
 		outputNode = graph.createNode( nodeData );
 		graph.addNode( outputNode );
		aig.getOutputNodes().push_back( outputNode );
		nodes[ nodeData.id ] = outputNode;
	}

	aig.setOutputNames( this->outputNames );
}

void AigParser::createAnds( aig::Aig & aig ) {
	aig::Graph & graph = aig.getGraph();
	for (int in = 0; in < a; in++) {
		// Node Data
		aig::NodeData nodeData;
		nodeData.id = ( std::get<0>( ands[in] ) / 2 );
		std::stringstream sName;
		// sName << "a" << in;
		// nodeData.name = sName.str();
		nodeData.counter = 0;
		nodeData.type = aig::NodeType::AND;
		// AIG Node
 		aig::GNode andNode;
 		andNode = graph.createNode( nodeData );
 		graph.addNode( andNode );
		nodes[ nodeData.id ] = andNode;
	}
}

void AigParser::connectLatches( aig::Aig & aig ) {
	aig::Graph & graph = aig.getGraph();
	for (int in = 0; in < l; in++) {
		int lhs = std::get<0>( latches[in] );
		aig::GNode latchNode = nodes[lhs / 2];
		int rhs = std::get<1>( latches[in] );
		aig::GNode inputNode = nodes[rhs / 2];
		graph.getEdgeData( graph.addEdge( inputNode, latchNode ) ) = !(rhs % 2);;
	}
}

void AigParser::connectOutputs( aig::Aig & aig ) {
	aig::Graph & graph = aig.getGraph();
	for (int in = 0; in < o; in++) {
		aig::GNode outputNode = nodes[m + in + 1];
		int rhs = outputs[in];
		aig::GNode inputNode = nodes[rhs / 2];
		graph.getEdgeData( graph.addEdge( inputNode, outputNode ) ) = !(rhs % 2);
	}
}

void AigParser::connectAnds( aig::Aig & aig ) {

	aig::Graph & graph = aig.getGraph();

	for ( auto andDef : this->ands ) {

		int lhs = std::get<0>( andDef );
		aig::GNode andNode = nodes[lhs / 2];
		int rhs0 = std::get<1>( andDef );
		aig::GNode inNode0 = nodes[rhs0 / 2];
		int rhs1 = std::get<2>( andDef );
		aig::GNode inNode1 = nodes[rhs1 / 2];

		graph.getEdgeData( graph.addMultiEdge( inNode0, andNode, Galois::MethodFlag::UNPROTECTED ) ) = !(rhs0 % 2);
		graph.getEdgeData( graph.addMultiEdge( inNode1, andNode, Galois::MethodFlag::UNPROTECTED ) ) = !(rhs1 % 2);
	}
}

int AigParser::getI() {
	return i;
}

int AigParser::getL() {
	return l;
}

int AigParser::getO() {
	return o;
}

int AigParser::getA() {
	return a;
}

int AigParser::getE( aig::Aig & aig ) {

	aig::Graph & graph = aig.getGraph();
	int nEdges = 0;

	for ( auto node : graph ) {
		nEdges += std::distance( graph.edge_begin( node ), graph.edge_end( node ) );
	}
	
	return nEdges;
}
AigParser::~AigParser() {
	close();
}
