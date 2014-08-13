#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include "MatrixGeneration/GenericMatrixGenerator.hxx"
#include "Point2D/MatrixGenerator.hxx"
#include "Point3D/MatrixGenerator.hxx"
#include "Edge2D/MatrixGenerator.hxx"
#include "FakeMatrixGenerator.h"

#include "EquationSystem.h"
#include "GaloisWorker.h"
#include "mpi.h"
#include "dmumps_c.h"
#include <map>
#include <sys/time.h>
#include "TaskDescription.h"
#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654


int call_mumps(int argc, char ** argv, int* in, int* jn, double* a, double* rhs, int n, int nz) {


  DMUMPS_STRUC_C id;
 
  int myid, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // Initialize a MUMPS instance. Use MPI_COMM_WORLD.
  printf("UNSYMETIC MATRIX\n");
  id.job=JOB_INIT; id.par=1; id.sym=1;
  id.comm_fortran=USE_COMM_WORLD;
  dmumps_c(&id);
  // Define the problem on the host
  if (myid == 0) {
     id.n = n; id.nz =nz; id.irn=in; id.jcn=jn;
     id.a = a; id.rhs = rhs;
  }

#define ICNTL(I) icntl[(I)-1]

  id.ICNTL(1)=0; id.ICNTL(2)=0; id.ICNTL(3)=0; id.ICNTL(4)=0; id.ICNTL(7)=2;
  // Outputs for debugging
  //printf("debugging\n"); 
  //id.ICNTL(1)=6; id.ICNTL(2)=6; id.ICNTL(3)=6; id.ICNTL(4)=4;
  // Call the MUMPS package.
  //id.job=6;

   timeval start_time;
   timeval end_time;

   id.job = 1; 
   gettimeofday(&start_time, NULL);
   dmumps_c(&id);
   gettimeofday(&end_time, NULL);
   printf("Mumps time %f [s]\n", ((end_time.tv_sec - start_time.tv_sec)*1000000 +(end_time.tv_usec - start_time.tv_usec))/1000000.0);
   id.job = 2; 
   gettimeofday(&start_time, NULL);
   dmumps_c(&id);
   gettimeofday(&end_time, NULL);
   printf("Mumps time %f [s]\n", ((end_time.tv_sec - start_time.tv_sec)*1000000 +(end_time.tv_usec - start_time.tv_usec))/1000000.0);
   id.job = 3; 
   gettimeofday(&start_time, NULL);
   dmumps_c(&id);
   gettimeofday(&end_time, NULL);
   printf("Mumps time %f [s]\n", ((end_time.tv_sec - start_time.tv_sec)*1000000 +(end_time.tv_usec - start_time.tv_usec))/1000000.0);
   

  id.job=JOB_END; dmumps_c(&id); // Terminate instance
  return 0;
}

int execute_mumps(int argc, char** argv, TaskDescription &taskDescription){

	printf("You are executing sequential mumps\n");
	GenericMatrixGenerator* matrix_generator;
        if(taskDescription.dimensions == 3) {
                switch (taskDescription.singularity) {
               	case POINT:
                       	matrix_generator = new D3::MatrixGenerator();
                        break;
                case CENTRAL_POINT:
                       	matrix_generator = new PointCentral3DMatrixGenerator();
                        break;
                case EDGE:
                       	matrix_generator = new Edge3DMatrixGenerator();
                        break;
                case FACE:
                        matrix_generator = new Face3DMatrixGenerator();
                        break;
               	case ANISOTROPIC:
                       	matrix_generator = new Anisotropic3DMatrixGenerator();
                        break;

               	}
        }
        else if (taskDescription.dimensions == 2 && !taskDescription.quad) {
                switch (taskDescription.singularity) {
               	case POINT:
                       	matrix_generator = new D2::MatrixGenerator();
                       	break;
                case CENTRAL_POINT:
                       	matrix_generator = new PointCentral2DMatrixGenerator();
                        break;
                case EDGE:
                        matrix_generator = new D2Edge::MatrixGenerator();
                        break;
                default:
                        printf("Error: unknown type of singularity in 2D!\n");
                        exit(1);
                        break;
                }
	}

	std::vector<EquationSystem*>* tier_vector = matrix_generator->CreateMatrixAndRhs(taskDescription);
	srand(time(NULL));
	double** matrix = matrix_generator->GetMatrix();
	double* rhs = matrix_generator->GetRhs();
	int size = matrix_generator->GetMatrixSize();

	int* in;
	int* jn;
	double* a;
	int nz;

	if(matrix != NULL)
	{
		//0 denotes no support of 2 basis function product, thats why it's possible to compare value to 0
		size = matrix_generator->GetMatrixSize();
		nz = 0;
		for(int i = 0; i<size; i++)
		{
			for(int j = i; j<size; j++)
				if(matrix[i][j] != 0)
					nz++;
		}

		in = (int*)malloc(sizeof(int) * nz);
		jn = (int*)malloc(sizeof(int) * nz);
		a = (double*)malloc(sizeof(double) * nz);
		int matrix_entry_nr = 0;
		for(int i = 0; i<size; i++)
			for(int j = 0; j<size; j++)
				if(matrix[i][j] != 0 && i>=j)
				{
					in[matrix_entry_nr] = i + 1;
					jn[matrix_entry_nr] = j + 1;
					a[matrix_entry_nr] = matrix[i][j];
					matrix_entry_nr++;
				}
	}
	else
	{
		if(taskDescription.dimensions == 2)
		{
			matrix_generator->GetMumpsArrays(in,jn,a,rhs,size,nz);

			for(int i = 0; i<nz; i++)
			{
				in[i]++; jn[i]++;
			}
		}

	}

    call_mumps(argc,argv,in,jn,a,rhs,size,nz);

    std::map<int,double>* solution_map = new std::map<int,double>();
    for(int i = 0; i<size; i++)
    	(*solution_map)[i] = rhs[i];
    matrix_generator->checkSolution(solution_map,taskDescription.function);

    return 0;

}
