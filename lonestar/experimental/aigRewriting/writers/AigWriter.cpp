#include "AigWriter.h"
#include "../util/utilString.h"

AigWriter::AigWriter() {
}

AigWriter::AigWriter( std::string path ) {
	this->path = path;
	setFile(path);
}

AigWriter::~AigWriter() {
	aigerFile.close();
}

void AigWriter::setFile( std::string path ) {
	this->path = path;
	aigerFile.close();
	aigerFile.open(path.c_str(), std::ios::trunc);
}

bool AigWriter::isOpen() {
	return aigerFile.is_open();
}

void AigWriter::writeAag( Aig & aig ) {
	aig.resetAndIDs();
	writeAagHeader( aig );
	writeInputs( aig );
	writeLatchesAag( aig );
	writeOutputs( aig );
	writeAndsAag( aig );
	writeSymbolTable( aig );
}

void AigWriter::writeAagHeader( Aig & aig ) {
	int i = aig.getNumInputs();
	int l = aig.getNumLatches();
	int o = aig.getNumOutputs();
	int a = aig.getNumAnds();
	int m = i + l + a;
	aigerFile << "aag " << m << " " << i << " " << l << " " << o << " " << a << std::endl;
}

void AigWriter::writeInputs( Aig & aig ) {

	aig::Graph & graph = aig.getGraph();

	for ( auto input : aig.getInputNodes() ) {
		aig::NodeData & inputData = graph.getData( input, galois::MethodFlag::READ );
		aigerFile << inputData.id * 2 << std::endl;
	}
}

void AigWriter::writeLatchesAag( Aig & aig ) {
	
	//FIXME

	/*
	auto latchNodes = aig.getLatchNodes();
	for (auto latch : latchNodes) {
		aig::NodeLatch* latchNode = latch.second;
		aigerFile << nodeIds[latchNode->getId()] * 2 << " ";
		aig::Node* input = latchNode->getInputNodes().front();
		bool polarity = latchNode->getInputPolarities().front();
		bool init = latchNode->getInitialValue();
		int inputId = 2 * nodeIds[input->getId()];
		inputId = polarity ? inputId : inputId + 1;
		aigerFile << inputId << " " << init << std::endl;
	}
	*/
}

void AigWriter::writeOutputs( Aig & aig ) {
	
	aig::Graph & graph = aig.getGraph();

	for ( auto output : aig.getOutputNodes() ) {

		auto inEdge = graph.in_edges( output ).begin();
		bool inEdgePolarity = graph.getEdgeData( *inEdge, galois::MethodFlag::READ );
		aig::GNode inNode = graph.getEdgeDst( *inEdge );
		aig::NodeData & inNodeData = graph.getData( inNode, galois::MethodFlag::READ );
		if ( inEdgePolarity ) {
			aigerFile << inNodeData.id * 2 << std::endl;
		}
		else {
			aigerFile << (inNodeData.id * 2) + 1 << std::endl;		
		}
	}
}

void AigWriter::writeAndsAag( Aig & aig ) {

	std::stack< aig::GNode > stack;

	aig.computeTopologicalSort( stack );

	aig::Graph & graph = aig.getGraph();

	unsigned int currentID = aig.getNumInputs() + aig.getNumLatches() + 1;

	while ( !stack.empty() ) {

		aig::GNode node = stack.top();
		stack.pop();

		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		//std::cout << nodeData.id << " -> ";
		//nodeData.id = currentID++; // Redefines the AND IDs according to the topological sorting.	
		//std::cout << nodeData.id << std::endl;		

		unsigned int andIndex = nodeData.id * 2;

		auto inEdge = graph.in_edges( node ).begin();
		bool lhsPolarity = graph.getEdgeData( *inEdge, galois::MethodFlag::READ );
		aig::GNode lhsNode = graph.getEdgeDst( *inEdge );
		aig::NodeData & lhsNodeData = graph.getData( lhsNode, galois::MethodFlag::READ );
		unsigned int lhsIndex = lhsNodeData.id * 2;
		lhsIndex = lhsPolarity ? lhsIndex : (lhsIndex+1);

		inEdge++;
		bool rhsPolarity = graph.getEdgeData( *inEdge, galois::MethodFlag::READ );
		aig::GNode rhsNode = graph.getEdgeDst( *inEdge );
		aig::NodeData & rhsNodeData = graph.getData( rhsNode, galois::MethodFlag::READ );
		unsigned int rhsIndex = rhsNodeData.id * 2;		
		rhsIndex = rhsPolarity ? rhsIndex : (rhsIndex+1); 
		
		if ( lhsIndex < rhsIndex ) {
			std::swap( lhsIndex, rhsIndex );
		}

		aigerFile << andIndex << " " << lhsIndex << " " << rhsIndex << std::endl;
	}
}

void AigWriter::writeAig( Aig & aig )  {
	aig.resetAndIDs();
	writeAigHeader( aig );
	writeLatchesAig( aig );
	writeOutputs( aig );
	writeAndsAig( aig );
	writeSymbolTable( aig );
}

void AigWriter::writeAigHeader( Aig & aig ) {
	int i = aig.getNumInputs();
	int l = aig.getNumLatches();
	int o = aig.getNumOutputs();
	int a = aig.getNumAnds();
	int m = i + l + a;
	aigerFile << "aig " << m << " " << i << " " << l << " " << o << " " << a << std::endl;
}

void AigWriter::writeLatchesAig( Aig & aig ) {
/*
	auto latchNodes = aig.getLatchNodes();
	for (auto latch : latchNodes) {
		aig::NodeLatch* latchNode = latch.second;
		aig::Node* input = latchNode->getInputNodes().front();
		bool polarity = latchNode->getInputPolarities().front();
		bool init = latchNode->getInitialValue();
		int inputId = 2 * nodeIds[input->getId()];
		inputId = polarity ? inputId : inputId + 1;
		aigerFile << inputId << " " << init << std::endl;
	}
*/

}

void AigWriter::writeAndsAig( Aig & aig ) {

	std::stack< aig::GNode > stack;

	aig.computeTopologicalSort( stack );

	aig::Graph & graph = aig.getGraph();

	unsigned int currentID = aig.getNumInputs() + aig.getNumLatches() + 1;

	while ( !stack.empty() ) {

		aig::GNode node = stack.top();
		stack.pop();

		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		//std::cout << nodeData.id << " -> ";
		//nodeData.id = currentID++; // Redefines the AND IDs according to the topological sorting.
		//std::cout << nodeData.id << std::endl;

		unsigned int andIndex = nodeData.id * 2;

		auto inEdge = graph.in_edges( node ).begin();
		bool lhsPolarity = graph.getEdgeData( *inEdge, galois::MethodFlag::READ );
		aig::GNode lhsNode = graph.getEdgeDst( *inEdge );
		aig::NodeData & lhsNodeData = graph.getData( lhsNode, galois::MethodFlag::READ );
		unsigned int lhsIndex = lhsNodeData.id * 2;
		lhsIndex = lhsPolarity ? lhsIndex : (lhsIndex+1);

		inEdge++;
		bool rhsPolarity = graph.getEdgeData( *inEdge, galois::MethodFlag::READ );
		aig::GNode rhsNode = graph.getEdgeDst( *inEdge );
		aig::NodeData & rhsNodeData = graph.getData( rhsNode, galois::MethodFlag::READ );
		unsigned int rhsIndex = rhsNodeData.id * 2;		
		rhsIndex = rhsPolarity ? rhsIndex : (rhsIndex+1); 
		
		if ( lhsIndex < rhsIndex ) {
			std::swap( lhsIndex, rhsIndex );
		}
	
		encode( andIndex - lhsIndex );
		encode( lhsIndex - rhsIndex );
	}
}

void AigWriter::writeSymbolTable( Aig & aig ) {
	
	int i = 0;
	for ( auto inputName : aig.getInputNames() ) {
		aigerFile << "i" << i++ << " " << inputName << std::endl;
	}

	i = 0;
	for ( auto latchName : aig.getLatchNames() ) {
		aigerFile << "l" << i++ << " " << latchName << std::endl;
	}

	i = 0;
	for ( auto outputName : aig.getOutputNames() ) {
		aigerFile << "o" << i++ << " " << outputName << std::endl;
	}

	aigerFile << "c" << std::endl << aig.getDesignName() << std::endl;
}

void AigWriter::encode( unsigned x ) {
	unsigned char ch;
	while (x & ~0x7f) {
		ch = (x & 0x7f) | 0x80;
		aigerFile.put(ch);
		x >>= 7;
	}
	ch = x;
	aigerFile.put(ch);
}

void AigWriter::close() {
	aigerFile.close();
}
