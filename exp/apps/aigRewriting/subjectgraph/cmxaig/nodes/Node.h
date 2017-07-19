#ifndef CMXAIG_NODE_H
#define CMXAIG_NODE_H

#include <string>
#include <vector>

namespace SubjectGraph {

typedef std::string String;
typedef std::vector<bool> EdgeValues;

/**
 * This class represents the Node basic data structure. It is the base class of
 * every node in the Cmxaig.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see Cmxaig.
 */
class Node {

	unsigned id; /**< The unique id among the nodes of a Cmxaig. */

	unsigned typeId; /**< The unique id among the AndNodes of a Cmxaig. */

	String name; /**< The name of node is defined according to the node type, e.g. And_1 */

	std::vector<Node*> inNodes;

	EdgeValues inEdges;

	std::vector<Node*> outNodes;

	EdgeValues outEdges;

public:

	/**
	 * Empty constructor.
	 */
	Node();

	/**
	 * Default constructor.
	 * @param id The unique id among the nodes of a Cmxaig.
	 * @param type_id The unique id among the nodes of a same type in a Cmxaig.
	 */
	Node(const unsigned & id, const unsigned & typeId);

	/**
	 * Copy constructor.
	 */
	Node(const Node & node);

	/**
	 * Operator= overload.
	 */
	Node& operator=(const Node & rhs);

	virtual ~Node();

	virtual void addInNode(Node* inNode, const bool& inPolatiry) = 0;

	virtual void addOutNode(Node* outNode, const bool& outPolatiry) = 0;

	virtual void removeInNode(Node* inNode) = 0;

	virtual void removeOutNode(Node* outNode) = 0;

	virtual bool isAndNode() const = 0;

	virtual bool isXorNode() const = 0;

	virtual bool isMuxNode() const = 0;

	virtual bool isChoiceNode() const = 0;

	virtual bool isFunctionNode() const = 0;

	virtual bool isInputNode() const = 0;

	virtual bool isOutputNode() const = 0;

	virtual String toDot() const = 0;


	/****************** GETTERS AND SETTERS ******************/

	/**
	 * Returns the id of the node.
	 * @return the id of the node.
	 */
	unsigned getId() const;

	/**
	 * Returns the type_id of the node.
	 * @return the type_id of the node.
	 */
	unsigned getTypeId() const;

	/**
	 * Returns the name of the node.
	 * @return "And_<type_id>".
	 */

	std::vector<Node*>& getInNodes();

	void setInNodes(const std::vector<Node*>& inNodes);

	EdgeValues& getInEdges();

	void setInEdges(const EdgeValues& inEdges);

	std::vector<Node*>& getOutNodes();

	void setOutNodes(const std::vector<Node*>& outNodes);

	EdgeValues& getOutEdges();

	void setOutEdges(const EdgeValues& outEdges);

	const String& getName() const;

	void setName(const String& name);

};

}  // namespace SubjectGraph

#endif
