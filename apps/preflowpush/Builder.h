#include "iostream"
#include "fstream"
#include "string"
#include"map"
#include<string.h>

using namespace std;

class Builder{


	class Pair{
		public:
			int i,j;
			Pair(int a,int b)
			{ 
				this->i=a;
				this->j=b;
			}
	};



	map<Pair*,int> edge_map;


	public:
	void read(Graph* b, string filename) {

		int width,height,dummy;
		char str[100];
		ifstream scanner;
		scanner.open(filename.c_str());
		scanner >>width;
		scanner >> height;
		scanner >> dummy;

		cout<<"****************************"<<endl;
		cout<<"Width of graph is : "<<width<<endl;
		cout<<"Height of graph is : "<<width<<endl;
		cout<<"****************************"<<endl;

		scanner.getline(str,100);

		numNodes=width*height;
		int sourceId=numNodes++;
		int sinkId=numNodes++;

		GNode nodes[numNodes];
		for (int i = 0; i < numNodes; i++)
		{
			Node n(i);
			if (i == sourceId)
				n.setSource(numNodes);
			if (i == sinkId)
				n.setSink();
			nodes[i] = b->createNode(n);
			b->addNode(nodes[i]);
		}

		source=nodes[sourceId];
		sink=nodes[sinkId];
		char* tok;
		scanner.getline(str,100);
		while(str!= NULL && strstr(str,"N: ")!=NULL )
		{
			tok=strtok(str," \n");
			int src=atoi(tok);
			tok=strtok(NULL," \n");
			int cap=atoi(tok);

			if (cap > 0) 
			{
				addEdge(b,source, nodes[src], cap, 0);
			} 
			else 
			{      
				addEdge(b,nodes[src], sink, -cap, 0);
			}

			while(scanner.getline(str,100)!=NULL)
			{
				if(strstr(str,"->")==NULL)
					break;

				char* tok2;
				tok2=strtok(str," \n");

				tok2=strtok(NULL," \n");
				int dst=atoi(tok2);
				tok2=strtok(NULL," \n");
				int scap=atoi(tok2);
				tok2=strtok(NULL," \n");
				int dcap=atoi(tok2);

				Node& e1=nodes[src].getData();
				Node& e2=nodes[dst].getData();
				Pair *p=new Pair(e1.id,e2.id);
				if(   edge_map.find(p)  ==  edge_map.end()   )
				{
					addEdge(b,nodes[src],nodes[dst],scap,dcap);
				}
			}

		}
		cout<<endl<<endl<<"Graph read sucessful"<<endl;
		cout<<endl<<endl<<"Processing Graph ...."<<endl;

		scanner.close();
	}

	//addEdge function implementation.....  

	void addEdge(Graph* b, GNode& src, GNode& dst, int scap, int dcap)
	{


		Node& n1=src.getData();
		Node& n2=dst.getData();

		Edge e1(n1.id,n2.id,scap);
		Edge e2(n2.id,n1.id,dcap);


		Pair *x=new Pair(n1.id,n2.id);
		Pair *y=new Pair(n2.id,n1.id);
		if (edge_map.find(x)!=edge_map.end())
		{
			(b->getEdgeData(src,dst)).addOriginalCapacity(scap);
		}
		else
		{
			b->addEdge(src,dst,e1);
			numEdges++;
			edge_map[x]=1;
		}

		if (edge_map.find(y)!=edge_map.end())
		{
			(b->getEdgeData(dst,src)).addOriginalCapacity(dcap);
		}
		else
		{
			b->addEdge(dst,src,e2);
			numEdges++;
			edge_map[y]=1;
		}
	}


};
