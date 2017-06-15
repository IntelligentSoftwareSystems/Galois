#ifndef CMXAIG_OUTPUT_NODE_H
#define CMXAIG_OUTPUT_NODE_H

#include <string>
#include <vector>

#include "Node.h"

namespace SubjectGraph {

/**
 * This class represents the OutputNode basic data structure. It represents
 * an output of the circuit represented by a Cmxaig.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Node.
 */
class OutputNode: public Node {

public:

	/**
	 * Empty constructor.
	 */
	OutputNode();

	/**
	 * Constructor without name. The name will be set to "Output_<type_id>".
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param type_id The unique id among the InputNodes of a Cmxaig.
	 */
	OutputNode(const unsigned & id, const unsigned & typeId);

	/**
	 * Constructor with name.
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param type_id The unique id among the InputNodes of a Cmxaig.
	 * @param name The name of the Cmxaig circuit input.
	 */
	OutputNode(const unsigned & id, const unsigned & typeId, const String & name);

	/**
	 * Copy constructor.
	 */
	OutputNode(const OutputNode & node);

	/**
	 * Operator= overload.
	 */
	OutputNode& operator=(const OutputNode & rhs);

	/**
	 * Adds an input-node to this node.
	 * @param in_node A pointer to the node to be added.
	 * @param position The position in the input list the node will be added.
	 * @param val The value of the edge (true or false).
	 */
	void addInNode(Node* inNode, const bool & inPolarity);

	/**
	 * Adds an output-node to this node.
	 * @param out_node A pointer to the node to be added.
	 * @param val The value of the edge (true or false).
	 */
	void addOutNode(Node* outNode, const bool & outPolarity);

	void removeInNode(Node* inNode);

	void removeOutNode(Node* outNode);

	/**
	 * Function used to test if this node is an AndNode.
	 * @return false.
	 */
	bool isAndNode() const;

	/**
	 * Function used to test if this node is a XorNode.
	 * @return false.
	 */
	bool isXorNode() const;

	/**
	 * Function used to test if this node is a MuxNode.
	 * @return false.
	 */
	bool isMuxNode() const;

	/**
	 * Function used to test if this node is a ChoiceNode.
	 * @return false.
	 */
	bool isChoiceNode() const;

	/**
	 * Function used to test if this node is a FunctionNode.
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
	 * @return true.
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
