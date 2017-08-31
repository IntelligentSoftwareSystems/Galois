#ifndef CMXAIG_MUX_NODE_H
#define CMXAIG_MUX_NODE_H

#include "Node.h"

namespace SubjectGraph {

/**
 * This class represents the MuxNode basic data structure. It represents a Mux
 * gate in the circuit represented by a Cmxaig.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Node, ThreeInputOperatorNode.
 */
class MuxNode: public Node {

public:

	/**
	 * Empty constructor.
	 */
	MuxNode();

	/**
	 * Default constructor.
	 * @param id The unique id among the nodes of a Cmxaig
	 * @param type_id The unique id among the MuxNodes of a Cmxaig
	 */
	MuxNode(const unsigned & id, const unsigned & type_id);

	/**
	 * Copy constructor.
	 */
	MuxNode(const MuxNode & node);

	/**
	 * Operator= overload.
	 */
	MuxNode& operator=(const MuxNode & rhs);

	void addInNode(Node* in_node, const bool & isInvEdge);

	void addOutNode(Node* out_node, const bool & isInvEdge);

	void removeInNode(Node* in_node);

	void removeOutNode(Node* out_node);

	/**
	 * Function used to test if this node is an AndNode.
	 * @return false.
	 */
	bool isAndNode() const;

	/**
	 * Function used to test if this node is an XorNode.
	 * @return false.
	 */
	bool isXorNode() const;

	/**
	 * Function used to test if this node is a MuxNode.
	 * @return true.
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
