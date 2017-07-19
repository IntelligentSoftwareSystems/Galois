#ifndef CMXAIG_INPUT_NODE_H
#define CMXAIG_INPUT_NODE_H

#include "Node.h"
#include "FunctionNode.h"

namespace SubjectGraph {

/**
 * This class represents the InputNode basic data structure. A InputNode is an
 * Input of the circuit represented by a Cmxaig.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Node, TwoInputOperatorNode.
 */
class InputNode: public FunctionNode {

public:

	/**
	 * Empty constructor.
	 */
	InputNode();

	/**
	 * Constructor without name. The name will be set to "Input_<type_id>".
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param typeId The unique id among the InputNodes of a Cmxaig.
	 * @param function The elementary logic function tied to this input node.
	 */
	InputNode(const unsigned & id, const unsigned & typeId, word * function);

	/**
	 * Constructor with name.
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param typeId The unique id among the InputNodes of a Cmxaig.
	 * @param name The name of the input node.
	 * @param function The elementary logic function tied to this input node.
	 */
	InputNode(const unsigned & id, const unsigned & typeId, String name, word * function);

	/**
	 * Copy constructor.
	 */
	InputNode(const InputNode & node);

	/**
	 * Operator= overload.
	 */
	InputNode& operator=(const InputNode & rhs);

	/**
	 * Adds an input-node to this node.
	 * @param in_node A pointer to the node to be added.
	 * @param position The position in the input list the node will be added.
	 * @param val The value of the edge (true or false).
	 */
	void addInNode(Node* inNode, const bool & visInvEdge);

	/**
	 * Adds an output-node to this node.
	 * @param out_node A pointer to the node to be added.
	 * @param val The value of the edge (true or false).
	 */
	void addOutNode(Node* outNode, const bool & isInvEdge);

	void removeInNode(Node* inNode);

	void removeOutNode(Node* outNode);


	/**
	 * Function used to test if this node is a ChoiceNode.
	 * @return false.
	 */
	bool isChoiceNode() const;

	/**
	 * Function used to test if this node is an InputNode.
	 * @return true.
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
