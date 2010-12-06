class Node
{
	public:
		int current;
		double excess;	
		int id;
		bool isSink;
		bool isSource;
		int height;
		long int insert_time;
	public:
		Node(int n);

		void setSink();

		void setSource(int numNodes);
		
		friend class Builder;
};

Node::Node(int n)
{
	this->id = n;
	height = 1;
	excess = 0;
	current = 0;
	isSource=false;
	isSink=false;
	insert_time=0;
}

void Node::setSink()
{
	height = 1;
	isSink = true;
}

void Node::setSource(int numNodes)
{
	height = numNodes;
	isSource = true;
}

