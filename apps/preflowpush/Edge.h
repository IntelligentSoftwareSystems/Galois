class Edge
  {
	private:
	int src,dst;	
  int cap;
	int ocap;

	public:
	Edge(int source, int dest, int ocap)
	{
	    this->src=source;
	    this->dst=dest;
	  this->cap = ocap;
	  this->ocap = ocap;
	}
	
	Edge() {}

	void addOriginalCapacity(int delta)
	{
	  cap += delta;
	  ocap += delta;
	}
  };
