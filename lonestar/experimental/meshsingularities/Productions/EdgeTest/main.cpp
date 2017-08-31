#include "../EdgeProduction.h"
#include <stdio.h>

int main(int argc, char** argv)
{

	std::vector<int>* v;
	Vertex* vertexC1 = new Vertex(NULL,NULL,NULL,ROOT,9);
	Vertex* vertexC2 = new Vertex(NULL,NULL,NULL,ROOT,9);
	EquationSystem* inDataC1 = new EquationSystem(9);
	EquationSystem* inDataC2 = new EquationSystem(9);


	for(int i = 0; i<9; i++)
		for(int j = 0; j<9; j++)
		{
			inDataC1->matrix[i][j] = 1;
			inDataC2->matrix[i][j] = 2;
			if(i == 0 || j == 0)
			{
				inDataC1->matrix[i][j] = -100;
				inDataC2->matrix[i][j] = -100;
			}


		}

	EdgeProduction* copy = new EdgeProduction(v);
	copy->Copy(vertexC1,inDataC1);
	copy->Copy(vertexC2,inDataC2);
	Vertex* vertexMC = new Vertex(vertexC2,vertexC1,NULL,ROOT,11);


	EdgeProduction* emc = new EdgeProduction(v);

	emc->MC(vertexMC);
	printf("MC non leaf test------------------------------------------------\n");
	vertexMC->system->print();
	printf("MC non leaf test------------------------------------------------\n");

	Vertex* vertexC1L = new Vertex(NULL,NULL,NULL,ROOT,9);
	Vertex* vertexC2L = new Vertex(NULL,NULL,NULL,ROOT,9);
	EquationSystem* inDataC1L = new EquationSystem(9);
	EquationSystem* inDataC2L = new EquationSystem(9);


	for(int i = 0; i<9; i++)
		for(int j = 0; j<9; j++)
		{
			inDataC1L->matrix[i][j] = 1;
			inDataC2L->matrix[i][j] = 2;
			if(i == 0 || j == 0 || i == 1 || j == 1)
			{
				inDataC1L->matrix[i][j] = -100;
				inDataC2L->matrix[i][j] = -100;
			}


		}

	EdgeProduction* copyL = new EdgeProduction(v);
	copyL->Copy(vertexC1L,inDataC1L);
	copyL->Copy(vertexC2L,inDataC2L);
	Vertex* vertexMCL = new Vertex(vertexC2L,vertexC1L,NULL,ROOT,11);


	EdgeProduction* emcl = new EdgeProduction(v);
	emcl->MBLeaf(vertexMCL);
	printf("MCleaf test------------------------------------------------\n");
	vertexMCL->system->print();
	printf("MCleaf test------------------------------------------------\n");


	Vertex* vertexMD = new Vertex(vertexC2,vertexC1,NULL,ROOT,11);
	EdgeProduction* emd = new EdgeProduction(v);

	emd->MD(vertexMD);
	printf("MD test------------------------------------------------\n");
	vertexMD->system->print();
	printf("MD test------------------------------------------------\n");

	//mc always right
	Vertex* vertexMBC = new Vertex(vertexMCL,vertexMC,NULL,ROOT,14);
	emcl->MBC(vertexMBC,false);
	printf("MBC test------------------------------------------------\n");
	vertexMBC->system->print();
	printf("MBC test------------------------------------------------\n");


	Vertex* vertexX1 = new Vertex(NULL,NULL,NULL,ROOT,14);
	Vertex* vertexX2 = new Vertex(NULL,NULL,NULL,ROOT,14);
	EquationSystem* inDataX1 = new EquationSystem(14);
	EquationSystem* inDataX2 = new EquationSystem(14);


	for(int i = 0; i<14; i++)
		for(int j = 0; j<14; j++)
		{
			inDataX1->matrix[i][j] = 1;
			inDataX2->matrix[i][j] = 2;
			if(i == 0 || j == 0 || i == 1 || j == 1 || i == 2 || j == 2)
			{
				inDataX1->matrix[i][j] = -100;
				inDataX2->matrix[i][j] = -100;
			}


		}

	EdgeProduction* copyX = new EdgeProduction(v);
	copyX->Copy(vertexX1,inDataX1);
	copyX->Copy(vertexX2,inDataX2);
	Vertex* vertexMB = new Vertex(vertexX1,vertexX2,NULL,ROOT,17);


	EdgeProduction* mb = new EdgeProduction(v);

	mb->MB(vertexMB);
	printf("MB test------------------------------------------------\n");
	vertexMB->system->print();
	printf("MBtest------------------------------------------------\n");

	return 0;
}
