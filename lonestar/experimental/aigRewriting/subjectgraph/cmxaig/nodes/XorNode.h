#ifndef CMXAIG_XOR_NODE_H
#define CMXAIG_XOR_NODE_H

#include "Node.h"

namespace SubjectGraph {

/**
 * This class represents the XorNode basic data structure.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Node, TwoInputOperatorNode.
 */
class XorNode: public Node {

public:

	/**
	 * Empty constructor.
	 */
	XorNode();

	/**
	 * Default constructor.
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param type_id The unique id among the XorNodes of a Cmxaig.
	 */
	XorNode(const unsigned& id, const unsigned& typeId);

	/**
	 * Copy constructor.
	 */
	XorNode(const XorNode& node);

	/**
	 * Operator= overload.
	 */
	XorNode& operator=(const XorNode& rhs);

	void addInNode(Node* inNode, const bool& inPolatiry);

	void addOutNode(Node* outNode, const bool& outPolatiry);

	void removeInNode(Node* inNode);

	void removeOutNode(Node* outNode);

	/**
	 * Function used to test if this node is an AndNode.
	 * @return false.
	 */
	bool isAndNode() const;

	/**
	 * Function used to test if this node is an XorNode.
	 * @return true.
	 */
	bool isXorNode() const;

	/**
	 * Function used to test if this node is an MuxNode.
	 * @return false.
	 */
	bool isMuxNode() const;

	/**
	 * Function used to test if this node is an ChoiceNode.
	 * @return false.
	 */
	bool isChoiceNode() const;

	/**
	 * Function used to test if this node is an FunctionNode.
	 * @return false.
	 */
	bool isFunctionNode() const;

	/**
	 * Function used to test if this node is an InputNode.
	 * @return false.
	 */
	bool isInputNode() const;

	/**
	 * Function used to test if this node is an OutputNode.
	 * @return false.
	 */
	bool isOutputNode() const;

	/**
	 * Returns a string definition of the node in dot format.
	 * @return a string definition of the node in dot format.
	 * @see http://www.graphviz.org/
	 */
	String toDot() const;

};

}  // namespace SubjectGraph

#endif
