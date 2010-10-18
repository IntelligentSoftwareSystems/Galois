class Node
  {
	private:
	int current;
	double excess;
	int height;
	

	public:
	int id;
	bool isSink;
	bool isSource;
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
