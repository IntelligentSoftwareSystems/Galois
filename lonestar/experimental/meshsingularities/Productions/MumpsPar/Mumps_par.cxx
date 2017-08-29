/*
 *
 *  This file is part of MUMPS 4.10.0, built on Tue May 10 12:56:32 UTC 2011
 *
 */
/* Example program using the C interface to the
 * double real arithmetic version of MUMPS, dmumps_c.
 * We solve the system A x = RHS with
 *   A = diag(1 2) and RHS = [1 4]^T
 * Solution is [1 2]^T */
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include "../MatrixGeneration/GenericMatrixGenerator.hxx";
#include "../Point2D/MatrixGenerator.hxx";
#include "../Point3D/MatrixGenerator.hxx";
#include "../EquationSystem.h"
#include "../GaloisWorker.h"
#include "mpi.h"
#include "dmumps_c.h"
#include <map>
#include <sys/time.h>
#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

using namespace D3;

int call_mumps(int argc, char ** argv, int* in, int* jn, double* a, double* rhs, int n, int nz, double** matrix)
{
  DMUMPS_STRUC_C id;


  int myid, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);


  /* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
  id.job=JOB_INIT; id.par=1; id.sym=0;id.comm_fortran=USE_COMM_WORLD;
  dmumps_c(&id);
  /* Define the problem on the host */
  if (myid == 0) {
    id.n = n; id.nz =nz; id.irn=in; id.jcn=jn;
    id.a = a; id.rhs = rhs;
  }
#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
/* No outputs */
  id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;
/* Call the MUMPS package. */
  id.job=6;
  timeval start_time;
  timeval end_time;
  int x = gettimeofday(&start_time, NULL);
  dmumps_c(&id);
  x += gettimeofday(&end_time, NULL);
  if(x == 0)
  {
  	  printf("Mumps time %f [s]\n", ((end_time.tv_sec - start_time.tv_sec)*1000000 +(end_time.tv_usec - start_time.tv_usec))/1000000.0);
  }
  id.job=JOB_END; dmumps_c(&id); /* Terminate instance */
  ierr = MPI_Finalize();
  return 0;
}

int execute_mumps(int argc, char** argv, TaskDescription &taskDescription){

	GenericMatrixGenerator* matrix_generator;
	if(taskDescription.dimensions == 3)
		matrix_generator = new D3::MatrixGenerator();
	else if(taskDescription.dimensions == 2)
		matrix_generator = new D2::MatrixGenerator();

	std::list<EquationSystem*>* tier_list = matrix_generator->CreateMatrixAndRhs(taskDescription);
	srand(time(NULL));
	double** matrix = matrix_generator->GetMatrix();
	double* rhs = matrix_generator->GetRhs();
	int size = matrix_generator->GetMatrixSize();

	//0 denotes no support of 2 basis function product, thats why it's possible to compare value to 0
	int nz = 0;
	for(int i = 0; i<size; i++)
	{
		for(int j = 0; j<size; j++)
			if(matrix[i][j] != 0)
				nz++;
	}

    int* in = (int*)malloc(sizeof(int) * nz);
    int* jn = (int*)malloc(sizeof(int) * nz);
    double* a = (double*)malloc(sizeof(double) * nz);
    int matrix_entry_nr = 0;
    for(int i = 0; i<size; i++)
    	for(int j = 0; j<size; j++)
    		if(matrix[i][j] != 0)
    		{
    			in[matrix_entry_nr] = i + 1;
    			jn[matrix_entry_nr] = j + 1;
    			a[matrix_entry_nr] = matrix[i][j];
    			matrix_entry_nr++;
    		}

    call_mumps(argc,argv,in,jn,a,rhs,size,nz,matrix);
    std::map<int,double>* solution_map = new std::map<int,double>();
    for(int i = 0; i<size; i++)
    	(*solution_map)[i] = rhs[i];
    matrix_generator->checkSolution(solution_map,taskDescription.function);
    return 0;

}
