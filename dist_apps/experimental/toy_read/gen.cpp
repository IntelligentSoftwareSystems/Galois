/** Toy Reading -*- C++ -*-
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <galois/Timer.h>

/******************************************************************************/
/* Main */
/******************************************************************************/

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  printf("File to read: %s\n", argv[1]);

  char dummyBuf[1];

  MPI_File mpiFile;
  MPI_Status mpiStat;

  MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, 
                MPI_INFO_NULL, &mpiFile);

  MPI_Offset fileSize;
  MPI_File_get_size(mpiFile, &fileSize);
  printf("MPI says size is %llu\n", fileSize);
  uint64_t mpiBytesRead = 0;

  galois::Timer fileReadTimer;

  fileReadTimer.start();

  while (true) {
    MPI_File_read_at(mpiFile, mpiBytesRead, (void*)dummyBuf, 1, MPI_BYTE, &mpiStat); 

    if (mpiStat.MPI_ERROR != MPI_SUCCESS) {
      printf("MPI Err is %d\n", mpiStat.MPI_ERROR);
      break;
    } else {
      mpiBytesRead++;
      if (mpiBytesRead == (unsigned)fileSize) {
        break;
      }
    }
  }

  fileReadTimer.stop();
  printf("MPI Read %lu bytes in %f seconds (%f MBPS)\n", mpiBytesRead,
         fileReadTimer.get_usec()/1000000.0f, 
         mpiBytesRead / (float)fileReadTimer.get_usec()) ;


  MPI_File_close(&mpiFile);

  std::FILE* openFile = std::fopen(argv[1], "r");

  uint64_t bytesRead = 0;

  galois::Timer fileReadTimer2;

  fileReadTimer2.start();
  while (true) {
    int fileReadStatus = std::getc(openFile);

    if (fileReadStatus != EOF) {
      bytesRead++;
    } else {
      break;
    }
  }
  fileReadTimer2.stop();

  std::fclose(openFile);

  printf("Regular read: Read %lu bytes in %f seconds (%f MBPS)\n", bytesRead,
         fileReadTimer2.get_usec()/1000000.0f, 
         bytesRead / (float)fileReadTimer2.get_usec()) ;


  return 0;

}
