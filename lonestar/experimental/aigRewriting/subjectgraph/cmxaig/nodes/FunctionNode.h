/*
 * FunctionNode.h
 *
 *  Created on: Nov 7, 2014
 *      Author: mhbackes
 */

#ifndef FUNCTIONNODE_H_
#define FUNCTIONNODE_H_

#include "../../functional/FunctionHandler.h"
#include "Node.h"

namespace SubjectGraph {

typedef Functional::word word;

/**
 * This class represents the FunctionNode basic data structure. A FunctionNode
 * is a node in the Cmxaig that contains the Function value of the nodes below.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Node, FunctionNode.
 */
class FunctionNode: public Node {

	//TODO change value to pointer.
	word * value; /**< The boolean function that describes the function node */

public:

	/**
	 * Empty constructor.
	 */
	FunctionNode();

	/**
	 * Default constructor.
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param type_id The unique id among the InputNode or ChoiceNode of a
	 * Cmxaig.
	 * @param value The boolean function that describes this node.
	 */
	FunctionNode(const unsigned & id, const unsigned & type_id, word * value);

	/**
	 * Copy constructor.
	 */
	FunctionNode(const FunctionNode & node);

	/**
	 * Operator= overload.
	 */
	FunctionNode& operator=(const FunctionNode & rhs);

	/**
	 * Returns the boolean function that describes the node.
	 * @return the boolean function that describes the node.
	 */
	word * getFunction() const;

	word * getFunctionPtr();

	/**
	 * Function used to test if this node is an AndNode.
	 * @return true.
	 */
	bool isAndNode() const;

	/**
	 * Function used to test if this node is a XorNode.
	 * @return false.
	 */
	bool isXorNode() const;

	/**
	 * Function used to test if this node is an MuxNode.
	 * @return false.
	 */
	bool isMuxNode() const;

	/**
	 * Function used to test if this node is a FunctionNode.
	 * @return true.
	 */
	bool isFunctionNode() const;

	/**
	 * Function used to test if this node is an OutputNode.
	 * @return false.
	 */
	bool isOutputNode() const;

	/**
	 * Default destructor.
	 */
	virtual ~FunctionNode();
};

} /* namespace subjectGraph */

#endif /* FUNCTIONNODE_H_ */
