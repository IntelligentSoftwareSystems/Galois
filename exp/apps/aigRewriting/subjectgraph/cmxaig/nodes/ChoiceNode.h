#ifndef CMXAIG_CHOICE_NODE_H
#define CMXAIG_CHOICE_NODE_H

#include "Node.h"
#include "FunctionNode.h"

namespace SubjectGraph {

/**
 * This class represents the ChoiceNode basic data structure.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Node, FunctionNode.
 */
class ChoiceNode: public FunctionNode {

public:

	/*
	 * Empty constructor.
	 */
	ChoiceNode();

	/**
	 * Default constructor.
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param type_id The unique id among the ChoiceNode of a Cmxaig.
	 * @param value The boolean function that describes this node.
	 */
	ChoiceNode(const unsigned & id, const unsigned & typeId, word * function);

	/**
	 * Copy constructor.
	 */
	ChoiceNode(const ChoiceNode & node);

	/**
	 * Operator= overload.
	 */
	ChoiceNode& operator=(const ChoiceNode & rhs);

	/**
	 * Adds an input-node to this node.
	 * @param in_node A pointer to the node to be added.
	 * @param position The position in the input list the node will be added.
	 * @param val The value of the edge (true or false).
	 */
	void addInNode(Node* inNode, const bool& inPolatiry);

	/**
	 * Adds an output-node to this node.
	 * @param out_node A pointer to the node to be added.
	 * @param val The value of the edge (true or false).
	 */
	void addOutNode(Node* outNode, const bool& outPolatiry);

	void removeInNode(Node* inNode);

	void removeOutNode(Node* outNode);

	/**
	 * Function used to test if this node is a ChoiceNode.
	 * @return true.
	 */
	bool isChoiceNode() const;

	/**
	 * Function used to test if this node is an InputNode.
	 * @return false.
	 */
	bool isInputNode() const;

	/**
	 * Returns a string definition of the node in dot format.
	 * @return a string definition of the node in dot format.
	 * @see http://www.graphviz.org/
	 */
	String toDot() const;

};

}  // namespace SubjectGraph

#endif
