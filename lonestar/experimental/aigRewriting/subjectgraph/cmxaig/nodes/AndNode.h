#ifndef CMXAIG_AND_NODE_H
#define CMXAIG_AND_NODE_H

#include "Node.h"

namespace SubjectGraph {

/**
 * This class represents the AndNode basic data structure. The AndNode
 * represents the AND operation of its two input nodes.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Node, TwoInputOperatorNode.
 */
class AndNode: public Node {

public:

	/**
	 * Empty constructor.
	 */
	AndNode();

	/**
	 * Default constructor.
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param type_id The unique id among the AndNodes of a Cmxaig.
	 */
	AndNode(const unsigned& id, const unsigned& typeId);

	/**
	 * Copy constructor.
	 */
	AndNode(const AndNode& node);

	/**
	 * Operator= overload.
	 */
	AndNode& operator=(const AndNode& rhs);

	void addInNode(Node* inNode, const bool& inPolatiry);

	void addOutNode(Node* outNode, const bool& outPolatiry);

	void removeInNode(Node* inNode);

	void removeOutNode(Node* outNode);

	/**
	 * Function used to test if this node is an AndNode.
	 * @return true.
	 */
	bool isAndNode() const;

	/**
	 * Function used to test if this node is an XorNode.
	 * @return false.
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

