/**
 * This class represents the Mixer module.
 *
 * @author Marcos Henrique Backes.
 */

#ifndef MIXER_MIXER_H
#define MIXER_MIXER_H

#include <string>
#include <vector>
#include <sstream>
#include <utility>
#include <map>
#include <stdexcept>

#include "../subjectgraph/cmxaig/nodes/Node.h"
#include "../subjectgraph/cmxaig/nodes/MuxNode.h"
#include "../subjectgraph/cmxaig/nodes/XorNode.h"
#include "../subjectgraph/cmxaig/nodes/AndNode.h"
#include "../subjectgraph/cmxaig/nodes/ChoiceNode.h"
#include "../subjectgraph/cmxaig/nodes/InputNode.h"
#include "../subjectgraph/cmxaig/nodes/OutputNode.h"
#include "../subjectgraph/cmxaig/Cmxaig.h"
#include "../functional/FunctionHandler.h"

namespace SubjectGraph {

typedef Functional::word word;
typedef std::pair<unsigned, unsigned> PairKey;
typedef std::pair<unsigned, PairKey> TripleKey;
typedef std::map<TripleKey, MuxNode*> MuxNodeMap;
typedef std::map<PairKey, XorNode*> XorNodeMap;
typedef std::map<PairKey, AndNode*> AndNodeMap;
typedef std::unordered_map< word*, ChoiceNode*, Functional::FunctionHasher, Functional::FunctionComparator > ChoiceNodeMap;
typedef std::unordered_map< std::string, InputNode* > InputNodeMap;
typedef std::unordered_map< std::string, word* > FunctionMap;
typedef std::unordered_map< std::string, std::pair< word*, unsigned int > > StringFunctionMap;

class Mixer{

	int nVars;
	int nWords;
	int poolSize;
	int nBucketsSTL;

	Functional::BitVectorPool functionPool;
	std::vector< std::string > varSet;
	StringFunctionMap literals;

	Functional::FunctionHasher functionHasher;
	Functional::FunctionComparator functionComparator;

	MuxNodeMap muxTable; /**< Makes mux nodes unique */
	XorNodeMap xorTable; /**< Makes xor nodes unique */
	AndNodeMap andTable; /**< Makes and nodes unique */
	InputNodeMap inputTable; /**< Makes input nodes unique */
	ChoiceNodeMap choiceTable; /**< Makes choice nodes unique */

	Cmxaig * cmxaig;

	static const char END = '\0';
	static const char LP = '(';
	static const char RP = ')';
	static const char AND = '*';
	static const char OR = '+';
	static const char XOR = '^';
	static const char NOT = '!';
	static const char MUX = 'm';
	static const char COMMA = ',';

	void expr1( std::istringstream& equation, NodeInfo & nodeInfo );

	void expr2 (std::istringstream& equation, NodeInfo & nodeInfo );

	void term( std::istringstream& equation, NodeInfo & nodeInfo );

	void prim( std::istringstream& equation, NodeInfo & nodeInfo );

	void var( std::istringstream& equation, NodeInfo & nodeInfo );

	void newOrOldAnd( FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo );

	void newOrOldXor( FunctionNode* lhs, FunctionNode* rhs, bool lhsPolarity, bool rhsPolarity, NodeInfo & nodeInfo );

	void newOrOldMux( FunctionNode* lhs, FunctionNode* rhs, FunctionNode* sel, bool lhsPolarity, bool rhsPolarity, bool selPolarity, NodeInfo & nodeInfo );

	void newOrOldChoice( Node* inNode, bool inPolarity, word * choiceFunction, NodeInfo & nodeInfo );

	void newOrOldInput( const String& name, NodeInfo & nodeInfo );


public:

	Mixer( int nVars, int poolSize, std::vector< std::string > & varSet );

	/**
	 * Operator= overload.
	 */
	Mixer& operator=(Mixer rhs);

	/**
	 * Default destructor.
	 */
	~Mixer();


	/**
	 * Mixes an expression with the Cmxaig. If there is no output equivalent to
	 * this expression, a new output will be created.
	 *
	 * @param expression A string representing an expression.
	 *
	 * @see Cmxaig(const String& expression, const unsigned& num_var) for
	 * more information about the expression syntax.
	 */
	void mix(String expression);

	void mix( String expression, std::string outputName );

	void mixSubfunction(String expression);

	Cmxaig* getCmxaig();

	void setCmxaig(Cmxaig* cmxaig);

};

}  // namespace SubjectGraph

#endif
