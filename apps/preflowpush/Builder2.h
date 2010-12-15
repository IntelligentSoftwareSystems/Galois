#include "iostream"
#include "fstream"
#include "string"
#include"map"
#include<string.h>

using namespace std;

class Builder{


	public:
	void read_rand(Graph* b, string filename) {

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
			nodes[i] = (b->createNode(n));
			b->addNode((nodes[i]),Galois::Graph::NONE,0);
		}
		source=nodes[sourceId];
		sink=nodes[sinkId];
		char* tok;
		scanner.getline(str,100);
		while(str!= NULL && strstr(str,"N: ")!=NULL )
		{
			tok=strtok(str," \n");
			tok=strtok(NULL," \n");
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

				//Node& e1=nodes[src].getData(Galois::Graph::NONE);
				//Node& e2=nodes[dst].getData(Galois::Graph::NONE);
			
				if(   ! (nodes[src]).hasNeighbor(nodes[dst])   )
				{
					addEdge(b,nodes[src],nodes[dst],scap,dcap);
				}
			}

		}
		cout<<endl<<endl<<"Graph read sucessful... "<<endl;
		cout<<endl<<endl<<"Processing Graph ...."<<endl;

		scanner.close();
	}


	void read_wash(Graph* b, string filename)
	{
	   char str[100],dum[10];
	   ifstream scanner;
	   scanner.open(filename.c_str());
	   scanner.getline(str,100);
	   scanner.getline(str,100);
	   char *tok,*dummy;
	   scanner >>dum>>dum>>numNodes>>numEdges;
		cout<<numNodes<<numEdges<<endl;
	   numNodes+=1;
	   GNode nodes[numNodes];
	   for(int i=0;i<numNodes;i++)
	   {	
		Node n(i);           
                nodes[i] = (b->createNode(n));
                b->addNode(nodes[i],Galois::Graph::NONE,0);
	   }

	   scanner.getline(str,100);
	   scanner.getline(str,100);
	   while(strlen(str)>3 )
	   {                
	      if(strrchr(str,'n')!=NULL)
	      {
		 tok=strtok(str," \n");
		 tok=strtok(NULL," \n");
		 int id=atoi(tok);
		 dummy=strtok(NULL," \n");
		 if(strcmp(dummy,"s")==0)
		 {
		    source=nodes[id];
		    cout<<"Source is "<<id<<endl;
		    source.getData(Galois::Graph::NONE,0).setSource(numNodes);
		 }
		 else if(strcmp(dummy,"t")==0)
		 {
		    sink=nodes[id];
		    cout<<"Sink is "<<id<<endl;
		    sink.getData(Galois::Graph::NONE,0).setSink();
		 }
	      }
	      else if(strrchr(str,'a')!=NULL)
	      {
		 tok=strtok(str," \n");
		 tok=strtok(NULL," \n");
		 int src=atoi(tok);
		 tok=strtok(NULL," \n");
		 int dst=atoi(tok);
		 tok=strtok(NULL," \n");
		 int cap=atoi(tok);
		 addEdge(b,nodes[src],nodes[dst],cap,0);

	      }
	      scanner.getline(str,100);
	   }
	   scanner.close();
	}

	//addEdge function implementation.....  

	void addEdge(Graph* b, GNode& src, GNode& dst, int scap, int dcap)
	{


		Node& n1=src.getData(Galois::Graph::NONE,0);
		Node& n2=dst.getData(Galois::Graph::NONE,0);

		Edge e1(n1.id,n2.id,scap);
		Edge e2(n2.id,n1.id,dcap);
/*
		Pair *x=new Pair(n1.id,n2.id);
		Pair *y=new Pair(n2.id,n1.id);
		if (edge_map.find(x)!=edge_map.end())
		{
			(b->getEdgeData_directed(src,dst)).addOriginalCapacity(scap);
		}
		else
		{
			b->addEdge(src,dst,e1);
			edge_map[x]=1;
		}

		if (edge_map.find(y)!=edge_map.end())
		{
			(b->getEdgeData_directed(dst,src)).addOriginalCapacity(dcap);
		}
		else
		{
			b->addEdge(dst,src,e2);
			edge_map[y]=1;
		}*/

		if (src.hasNeighbor(dst)) {
		      b->getEdgeData(src, dst,Galois::Graph::NONE,0).addOriginalCapacity(scap);
    		} else {
      		      b->addEdge(src, dst,e1,Galois::Graph::NONE,0);
    		}

	        if (dst.hasNeighbor(src)) {
 		      b->getEdgeData(dst, src,Galois::Graph::NONE,0).addOriginalCapacity(dcap);
	        } else {
     		      b->addEdge(dst, src,e2,Galois::Graph::NONE,0);
    		}
	}


};
