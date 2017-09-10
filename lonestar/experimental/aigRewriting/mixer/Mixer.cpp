#include "Mixer.h"

#include <algorithm>
#include <iostream>

namespace SubjectGraph {

Mixer::Mixer( int nVars, int poolSize, std::vector< std::string > & varSet ) :
		nVars( nVars ),
		nWords( Functional::wordNum( nVars ) ),
		poolSize( poolSize ),
		nBucketsSTL( nVars * nWords * 100 ),
		functionPool( poolSize, nWords ),
		functionHasher( nWords ),
		functionComparator( nWords ),
		choiceTable( nBucketsSTL, functionHasher, functionComparator ) {

	this->varSet = varSet;
	Functional::createLiterals( this->varSet, this->literals, this->functionPool );

	this->cmxaig = new Cmxaig( this->nVars, this->nWords, this->functionPool );

	for( auto var : this->varSet ) {
		// Does not create a input node for constants 0 and 1.
		// It will be done, if necessary, during the mix process.
		NodeInfo nodeInfo;
		this->cmxaig->addInputNode( var, this->literals[ var ].first, nodeInfo );
		this->inputTable.insert( std::make_pair( var , ( (InputNode*) nodeInfo.node ) ) );
	}
}

Mixer::~Mixer() {
}

void Mixer::mix( String expression ) {

	expression.erase(std::remove_if(expression.begin(), expression.end(), isspace), expression.end());
	std::istringstream expressionStream(expression);

	NodeInfo nodeInfo;
	expr1(expressionStream, nodeInfo);
	ChoiceNode* choiceNode = (ChoiceNode*) nodeInfo.node;

	for(unsigned i=0; i < choiceNode->getOutNodes().size(); i++) {
		if( choiceNode->getOutNodes()[i]->isOutputNode() && (choiceNode->getOutEdges()[i] == nodeInfo.outPolarity) ) {
			return;
		}
	}

	this->cmxaig->addOutputNode(choiceNode, nodeInfo.outPolarity, nodeInfo );
}

void Mixer::mix( String expression, std::string outputName ) {

	expression.erase(std::remove_if(expression.begin(), expression.end(), isspace), expression.end());
	std::istringstream expressionStream(expression);

	NodeInfo nodeInfo;
	expr1(expressionStream, nodeInfo);
	ChoiceNode* choiceNode = (ChoiceNode*) nodeInfo.node;

	for(unsigned i=0; i < choiceNode->getOutNodes().size(); i++) {
		if( choiceNode->getOutNodes()[i]->isOutputNode() && (choiceNode->getOutEdges()[i] == nodeInfo.outPolarity) ) {
			return;
		}
	}

	this->cmxaig->addOutputNode(choiceNode, nodeInfo.outPolarity, outputName, nodeInfo );
}

void Mixer::mixSubfunction( String expression ) {

	expression.erase(std::remove_if(expression.begin(), expression.end(), isspace), expression.end());
	std::istringstream expressionStream(expression);

	NodeInfo nodeInfo;
	expr1( expressionStream, nodeInfo );
}

void Mixer::expr1 (std::istringstream & equation, NodeInfo & nodeInfo ) {

	expr2(equation, nodeInfo);

	char c = equation.peek();

	while (c == OR) {

		equation.ignore();

		FunctionNode* lhs = (FunctionNode*) nodeInfo.node;
		bool lhsPolarity = nodeInfo.outPolarity;

		expr2(equation, nodeInfo);
		FunctionNode* rhs = (FunctionNode*) nodeInfo.node;
		bool rhsPolarity = nodeInfo.outPolarity;

		/* (lhs + rhs) <--> !(!lhs * !rhs) */
		newOrOldAnd(lhs, rhs, !lhsPolarity, !rhsPolarity, nodeInfo);
		word * negFunction = functionPool.getMemory();
		Functional::NOT( negFunction, nodeInfo.function, this->nWords );
		nodeInfo.function = negFunction;
		nodeInfo.outPolarity =  !nodeInfo.outPolarity;

		if(equation.tellg() == -1) {
			return;
		}
		else {
			c = equation.peek();
		}
	}
}

void Mixer::expr2( std::istringstream& equation, NodeInfo & nodeInfo ) {

	term(equation, nodeInfo);

	char c = equation.peek();

	while (c == XOR) {

		equation.ignore();

		FunctionNode* lhs = (FunctionNode*) nodeInfo.node;
		bool lhsPolarity = nodeInfo.outPolarity;

		term(equation, nodeInfo);
		FunctionNode* rhs = (FunctionNode*) nodeInfo.node;
		bool rhsPolarity = nodeInfo.outPolarity;

		newOrOldXor(lhs, rhs, lhsPolarity, rhsPolarity, nodeInfo);

		if(equation.tellg() == -1) {
			return;
		}
		else {
			c = equation.peek();
		}
	}
}

void Mixer::term( std::istringstream& equation, NodeInfo & nodeInfo ) {

	prim(equation, nodeInfo);

	char c = equation.peek();

	while (c == AND) {

		equation.ignore();

		FunctionNode* lhs = (FunctionNode*) nodeInfo.node;
		bool lhsPolarity = nodeInfo.outPolarity;

		prim(equation, nodeInfo);
		FunctionNode* rhs = (FunctionNode*) nodeInfo.node;
		bool rhsPolarity = nodeInfo.outPolarity;

		newOrOldAnd(lhs, rhs, lhsPolarity, rhsPolarity, nodeInfo);

		if(equation.tellg() == -1) {
			return;
		}
		else {
			c = equation.peek();
		}
	}
}

void Mixer::prim( std::istringstream& equation, NodeInfo & nodeInfo ) {
	
	FunctionNode *lhs, *rhs, *sel;
	bool lhsPolarity, rhsPolarity, selPolarity;

	InputNode* inputNode;
	String name;
	std::pair<String, InputNode*> newEntry;

	char c = equation.peek();

	switch (c) {
		case NOT: {

			equation.ignore();

			prim(equation, nodeInfo);
			word * negFunction = functionPool.getMemory();
			Functional::NOT( negFunction, nodeInfo.function, this->nWords );
			nodeInfo.function = negFunction;
			nodeInfo.outPolarity = !nodeInfo.outPolarity;
			break;
		}

		case LP: {

			equation.ignore();

			expr1(equation, nodeInfo);

			c = equation.get();

			if (c != RP) {
				throw std::invalid_argument("Missing ')'.");
			}
			break;
		}

		case MUX: {

			equation.ignore();

			c = equation.get();

			if (c != LP) {
				throw std::invalid_argument("Missing '(' after m.");
			}
			expr1(equation, nodeInfo);
			lhs = (FunctionNode*) nodeInfo.node;
			lhsPolarity = nodeInfo.outPolarity;

			c = equation.get();
			if (c != COMMA) {
				throw std::invalid_argument("Missing ',' in m expression.");
			}
			expr1(equation, nodeInfo);
			rhs = (FunctionNode*) nodeInfo.node;
			rhsPolarity = nodeInfo.outPolarity;

			c = equation.get();
			if (c != COMMA) {
				throw std::invalid_argument("Missing ',' in m expression.");
			}
			expr1(equation, nodeInfo);
			sel = (FunctionNode*) nodeInfo.node;
			selPolarity =  nodeInfo.outPolarity;

			c = equation.get();
			if (c != RP) {
				throw std::invalid_argument("Missing ')' after m expression.");
			}

			newOrOldMux(lhs, rhs, sel, lhsPolarity, rhsPolarity, selPolarity, nodeInfo);
			break;
		}
/*
		case '0': {
			this->cmxaig->addConstantZero( this->literals["0"].first, nodeInfo );
			inputNode = (InputNode*) nodeInfo.node;
			name.push_back('0');
			newEntry = std::make_pair(name, inputNode);
			this->inputTable.insert(newEntry);
			equation.ignore();
			break;
		}

		case '1': {
			this->cmxaig->addConstantOne( this->literals["1"].first, nodeInfo );
			inputNode = (InputNode*) nodeInfo.node;
			name.push_back('1');
			newEntry = std::make_pair(name, inputNode);
			this->inputTable.insert(newEntry);
			equation.ignore();
			break;
		}
*/
		default:
			var(equation, nodeInfo);
	}
}

void Mixer::var( std::istringstream& equation, NodeInfo & nodeInfo ) {

	String name;
	const String delim_char("+^*!(),\0"); // "0" and "1" were removed form the delim_char
	char c = equation.peek();

	while (delim_char.find(c) == String::npos && (equation.tellg() != -1)) {
		name.push_back(c);
		equation.ignore();
		c = equation.peek();
	}

	newOrOldInput(name, nodeInfo);
}

void Mixer::newOrOldMux( FunctionNode* lhs, FunctionNode* rhs, FunctionNode* sel, bool lhsPolarity, bool rhsPolarity, bool selPolarity, NodeInfo & nodeInfo ) {

	PairKey pairKey;
	TripleKey tripleKey;
	MuxNode* muxNode;
	bool muxNodeOutPolarity;
	word * choiceFunction;
	ChoiceNode* choiceNode;
	unsigned first, second, third;

	if (lhs->getId() < rhs->getId() && rhs->getId() < sel->getId()) {
		first = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
		second = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
		third = selPolarity ? sel->getId()*2 : (sel->getId()*2)+1;
	}
	else {
		if (lhs->getId() < sel->getId() && sel->getId() < rhs->getId()) {
			first = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
			second = selPolarity ? sel->getId()*2 : (sel->getId()*2)+1;
			third = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
		}
		else {
			if (rhs->getId() < lhs->getId() && lhs->getId() < sel->getId()) {
				first = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
				second = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
				third = selPolarity ? sel->getId()*2 : (sel->getId()*2)+1;
			}
			else {
				if (rhs->getId() < sel->getId() && sel->getId() < lhs->getId()) {
					first = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
					second = selPolarity ? sel->getId()*2 : (sel->getId()*2)+1;
					third = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
				}
				else {
					if (sel->getId() < lhs->getId() && lhs->getId() < rhs->getId()) {
						first = selPolarity ? sel->getId()*2 : (sel->getId()*2)+1;
						second = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
						third = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
					}
					else {
						first = selPolarity ? sel->getId()*2 : (sel->getId()*2)+1;
						second = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
						third = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
					}
				}
			}
		}
	}

	pairKey = std::make_pair(second, third);
	tripleKey = std::make_pair(first, pairKey);

	MuxNodeMap::iterator muxIt = this->muxTable.find(tripleKey);

	if ( (muxIt != this->muxTable.end())  && (muxIt->second->getInEdges()[0] == lhsPolarity) && (muxIt->second->getInEdges()[1] == rhsPolarity) && (muxIt->second->getInEdges()[2] == selPolarity) ) {
		MuxNode* muxNode = muxIt->second;
		choiceNode = (ChoiceNode*) muxNode->getOutNodes()[0]; // Gets a pointer to the choice node connected to the output of the andNode
		choiceFunction = choiceNode->getFunctionPtr();
		muxNodeOutPolarity = muxNode->getOutEdges()[0];
		nodeInfo.node = choiceNode;
		nodeInfo.function = choiceFunction;
		nodeInfo.outPolarity = muxNodeOutPolarity;
	}
	else {
		this->cmxaig->addMuxNode(lhs, rhs, sel, lhsPolarity, rhsPolarity, selPolarity, nodeInfo);
		muxNode = (MuxNode*) nodeInfo.node;
		std::pair<TripleKey, MuxNode*> newEntry(tripleKey, muxNode);
		this->muxTable.insert(newEntry);
		muxNodeOutPolarity = true;
		choiceFunction = nodeInfo.function;
		newOrOldChoice(muxNode, muxNodeOutPolarity, choiceFunction, nodeInfo);
	}
}

void Mixer::newOrOldXor( FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo ) {

	PairKey pairKey;
	XorNode* xorNode;
	bool xorNodeOutPolarity;
	word * choiceFunction;
	ChoiceNode* choiceNode;
	unsigned first, second;

	if (lhs->getId() < rhs->getId()) {
		first = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
		second = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
	}
	else {
		first =  rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
		second = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
	}

	pairKey = std::make_pair(first, second);

	XorNodeMap::iterator xorIt = this->xorTable.find(pairKey);

	if ( (xorIt != this->xorTable.end()) ) { //&& (xorIt->second->getInEdges()[0] == lhsPolarity) && (xorIt->second->getInEdges()[1] == rhsPolarity) ) {
		xorNode = xorIt->second;
		choiceNode = (ChoiceNode*) xorNode->getOutNodes()[0]; // Gets a pointer to the choice node connected to the output of the andNode
		choiceFunction = choiceNode->getFunctionPtr();
		xorNodeOutPolarity = xorNode->getOutEdges()[0];
		nodeInfo.node = choiceNode;
		nodeInfo.function = choiceFunction;
		nodeInfo.outPolarity = xorNodeOutPolarity;
	}
	else {
		this->cmxaig->addXorNode(lhs, rhs, lhsPolarity, rhsPolarity, nodeInfo);
		xorNode = (XorNode*) nodeInfo.node;
		std::pair<PairKey, XorNode*> newEntry(pairKey, xorNode);
		this->xorTable.insert(newEntry);
		xorNodeOutPolarity = true;
		choiceFunction = nodeInfo.function;
		newOrOldChoice(xorNode, xorNodeOutPolarity, choiceFunction, nodeInfo);
	}
}

void Mixer::newOrOldAnd( FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo ) {

	PairKey pairKey;
	AndNode* andNode;
	bool andNodeOutPolarity;
	word * choiceFunction;
	ChoiceNode* choiceNode;
	unsigned first, second;

	if (lhs->getId() < rhs->getId()) {
		first = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
		second = rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
	}
	else {
		first =  rhsPolarity ? rhs->getId()*2 : (rhs->getId()*2)+1;
		second = lhsPolarity ? lhs->getId()*2 : (lhs->getId()*2)+1;
	}

	pairKey = std::make_pair(first, second);

	AndNodeMap::iterator andIt = this->andTable.find(pairKey);

	if ( (andIt != this->andTable.end()) ) { //&& (andIt->second->getInEdges()[0] == lhsPolarity) && (andIt->second->getInEdges()[1] == rhsPolarity) ) {
		andNode = andIt->second;
		choiceNode = (ChoiceNode*) andNode->getOutNodes()[0]; // Gets a pointer to the choice node connected to the output of the andNode
		choiceFunction = choiceNode->getFunctionPtr();
		andNodeOutPolarity = andNode->getOutEdges()[0];
		nodeInfo.node = choiceNode;
		nodeInfo.function = choiceFunction;
		nodeInfo.outPolarity = andNodeOutPolarity;
	}
	else {
		this->cmxaig->addAndNode(lhs, rhs, lhsPolarity, rhsPolarity, nodeInfo);
		andNode = (AndNode*) nodeInfo.node;
		std::pair<PairKey, AndNode*> newEntry(pairKey, andNode);
		this->andTable.insert(newEntry);
		andNodeOutPolarity = true;
		choiceFunction = nodeInfo.function;
		newOrOldChoice(andNode, andNodeOutPolarity, choiceFunction, nodeInfo);
	}
}

void Mixer::newOrOldChoice( Node* inNode, bool inPolarity, word * choiceFunction, NodeInfo & nodeInfo ) {

	ChoiceNode* choiceNode;
	word * f;

	if ( Functional::isOdd( choiceFunction ) ) {
		f = functionPool.getMemory();
		Functional::NOT( f, choiceFunction, this->nWords );
		inPolarity = !inPolarity;
	}
	else {
		f = choiceFunction;
	}

	ChoiceNodeMap::iterator choiceIt = this->choiceTable.find( f );

	if (choiceIt != this->choiceTable.end()) {
		choiceNode = (ChoiceNode*) choiceIt->second;
		cmxaig->addEdge(inNode, choiceNode, inPolarity);
		nodeInfo.node = choiceNode;
		nodeInfo.function = f;
		nodeInfo.outPolarity = inPolarity;

	}
	else {
		this->cmxaig->addChoiceNode(inNode, inPolarity, f, nodeInfo);
		choiceNode = (ChoiceNode*) nodeInfo.node;
		this->choiceTable.insert( std::make_pair(choiceNode->getFunctionPtr(), choiceNode) );
	}
}

void Mixer::newOrOldInput( const String& name, NodeInfo & nodeInfo ) {

	InputNodeMap::iterator it = this->inputTable.find(name);
	InputNode* inputNode;

	if (it != this->inputTable.end()) {
		inputNode = it->second;
		nodeInfo.node = inputNode;
		nodeInfo.function = inputNode->getFunctionPtr();
		nodeInfo.outPolarity = true;
	}
	else {
		auto it = this->literals.find(name);
		if( it != this->literals.end() ) {
			this->cmxaig->addInputNode( name, it->second.first, nodeInfo );
			inputNode = (InputNode*) nodeInfo.node;
			std::pair<String, InputNode*> newEntry(name, inputNode);
			this->inputTable.insert(newEntry);
		}
		else {
			throw std::invalid_argument( ("Input " + name + " was not found!") );
		}
	}
}

Cmxaig* Mixer::getCmxaig() {
	return this->cmxaig;
}

void Mixer::setCmxaig(Cmxaig* cmxaig) {
	this->cmxaig = cmxaig;
}

}  // namespace Subjectgraph
