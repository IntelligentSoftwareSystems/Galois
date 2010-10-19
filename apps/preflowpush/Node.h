class Node
  {
	public:
	int current;
	double excess;
	

	int id;
	bool isSink;
	bool isSource;
	int height;
	Node(int n)
	{
	  this->id = n;
	  height = 1;
	  excess = 0;
	  current = 0;
	}

	void setSink()
	{
	  height = 1;
	  isSink = true;
	}

	void setSource(int numNodes)
	{		
	  height = numNodes;
	  isSource = true;
	}
  };
