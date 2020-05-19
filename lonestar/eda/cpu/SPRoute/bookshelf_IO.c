/* -----------  FastPlace - Version 1.0 ----------------
                       by 
   Natarajan Viswanathan and Chris C.-N. Chu
     Dept. of ECpE, Iowa State University
          Copyright (c) - 2004 
Iowa State University Research Foundation, Inc.
--------------------------------------------------------*/
/* --------------------------------------------------------------------------
   Contains routines to:
   - Read and Write the benchmark files in Bookshelf format 
----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "memAlloc.h"
#include "bookshelf_IO.h"

#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define MIN(a,b) ((a)<(b) ? (a) : (b))


/*-- extern variables --*/
    // from createHash()
    char **cellName;
    
    // from readAuxFile()
    char nodesFile[BUFFERSIZE], netsFile[BUFFERSIZE], wtsFile[BUFFERSIZE]; 
    char sclFile[BUFFERSIZE], plFile[BUFFERSIZE], benchmarkName[BUFFERSIZE];
    
    // from readNodesFile()
    int movableNodes, numTerminals;
    float averageCellWidth, *cellWidth, *cellHeight; 
    
    // from readNetsFile() 
    int numPins, *netlist, *netlistIndex;
    float *xPinOffset, *yPinOffset;
       
    // from readPlFile()
    float *xCellCoord, *yCellCoord, minX, maxX, minY, maxY;
    int *areaArrayIO, numAreaArrayIO;


    // from readSclFile()
    int numRows, numRowBlockages;
    float siteOriginY, siteEndY, coreHeight;
    float siteOriginX, siteEndX, coreWidth;
    float siteWidth, siteSpacing, coreRowHeight;
    float *rowOriginX, *rowEndX;
    float *xRowBlockage, *yRowBlockage, *widthRowBlockage;

/*-- global variables --*/
    typedef struct nodesHash NODES;
    struct nodesHash  {
        char name[STRINGLEN];
        unsigned long index;
    };

    NODES *NodesInfo;

    long hashSize, hashBits, *RN;
    unsigned char *hashFlag;
    long modresNum;
    int numNodes;


/*-- functions --*/
    void createHash(char benchmarkPath[], char nodesFile[]);
    void freeHash();
    void readAuxFile(char benchmarkPath[], char auxFile[]);
    void readNodesFile(char benchmarkPath[], char nodesFile[]);
    void readNetsFile(char benchmarkPath[], char netsFile[]);
    void readPlFile(char benchmarkPath[], char plFile[]);
    void readSclFile(char benchmarkPath[], char sclFile[]);
    void writePlFile(char outputDir[], char benchmarkName[], float xCoord[], float yCoord[]);


/* -----------------------------------------------------------
   Reads the .nodes file and creates a hash linking cell name 
   to cell index for all the nodes in the circuit 
   (movable nodes + fixed nodes + I/O pads)
   
   creates extern vars:
      cellName[]
----------------------------------------------------------- */
void createHash(char benchmarkPath[], char nodesFile[]) 
{
    FILE *fp;
    char line[BUFFERSIZE], temp[BUFFERSIZE], s4[BUFFERSIZE];
    float nodeWidth, nodeHeight;
    long currentPos, j, k, nodeIndex, nodeNo;
    long R, nonpin_ptr, pin_ptr, hashfunc, RN_index; 


    strcpy(temp, benchmarkPath);
    strcat(temp, "/");
    strcat(temp, nodesFile);
    
    if((fp=fopen(temp, "r")) == NULL) {
        printf("Error in opening: %s \n", temp);
        exit(1);
    }
    
    // Reading first few lines 
    if(!fgets(temp, BUFFERSIZE, fp)) abort();
    do {
        currentPos = ftell(fp);
        if(!fgets(temp, BUFFERSIZE, fp)) abort();
    } while( (temp[0] == '#') || (strlen(temp) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  

    // getting numNodes and numTerminals
    if(fscanf(fp, "NumNodes\t:\t%d\n", &numNodes) != 1) abort();
    if(fscanf(fp, "NumTerminals\t:\t%d\n", &numTerminals) != 1) abort();

    // in case there are any more comments or blank lines before actual cell information
    do {
        currentPos = ftell(fp);
        if(!fgets(temp, BUFFERSIZE, fp)) abort();
    } while( (temp[0] == '#') || (strlen(temp) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  
    
    // defining hash variables
    hashBits = 3+(long)(log((double)numNodes)/log((double)2));
    hashSize = (long) pow(2, (double)hashBits);
    NodesInfo = (NODES *) malloc(hashSize*sizeof(NODES));
    RN = (long *) lvector(1, hashSize);
    hashFlag = cvector(1, hashSize);   

    // global vector giving inverse mapping b/w cell names and cell indexes
    cellName = cmatrix(1, numNodes, 1, STRINGLEN); 

     // initialize hash flags
    for(j=1;j<=hashSize;j++)
        hashFlag[j] = 0;

    // generate random sequence
    R = 1;
    for(j=1;j<=hashSize;j++) {
        R = (5*R)%hashSize;
        RN[j] = R/4;
    }
    modresNum = (hashBits+2)/3;

    nonpin_ptr = 1;                      // movable nodes start from 1
    pin_ptr = numNodes-numTerminals+1;   // fixed nodes start from movableNodes+1
    
    for(nodeNo=1;nodeNo<=numNodes;nodeNo++) {

        if(!fgets(line, BUFFERSIZE, fp)) abort();
        strcpy(s4, "");
        sscanf(line, "%s%f%f%s\n", temp, &nodeWidth, &nodeHeight, s4);

        if(strcmp(s4, "terminal")==0) {

            // create array to save cell name
            strcpy(cellName[pin_ptr], temp);

            // create a hash table for name searching
            hashfunc = 0;
            for(j=1;j<=IMIN(strlen(temp), modresNum);j++)
                hashfunc += ((long)temp[j-1]<<3*(j-1))%hashSize;
          
            hashfunc = hashfunc%hashSize;
            RN_index = 1;

            while(hashFlag[hashfunc]!=0 && RN_index<hashSize) {
                hashfunc = (hashfunc+RN[RN_index])%hashSize;
                RN_index++;
            }

            if (RN_index>=hashSize) {  
                printf("cannot fill in hash table\n");
                exit(1);
            }
          
            strcpy(NodesInfo[hashfunc].name, temp);
            NodesInfo[hashfunc].index = pin_ptr;
            hashFlag[hashfunc] = 1;
          
            pin_ptr++;
       
        } else {

            // create array to save cell name
            strcpy(cellName[nonpin_ptr], temp);
          
            // create a hash table for name searching
            hashfunc = 0;
            for(j=1;j<=IMIN(strlen(temp), modresNum);j++)
                hashfunc += ((long)temp[j-1]<<3*(j-1))%hashSize;
          
            hashfunc = hashfunc%hashSize;
            RN_index = 1;
          
            while(hashFlag[hashfunc]!=0 && RN_index<hashSize) {
                hashfunc = (hashfunc+RN[RN_index])%hashSize;
                RN_index++;
            }
          
            if (RN_index>=hashSize) {  
                printf("cannot fill in hash table\n");
                exit(1);
            }
          
            strcpy(NodesInfo[hashfunc].name, temp);
            NodesInfo[hashfunc].index = nonpin_ptr;
            hashFlag[hashfunc] = 1;

            nonpin_ptr++;
        }
    }

    fclose(fp);  
}

    
/* -----------------------------------------------------------
  frees hash elements
----------------------------------------------------------- */
void freeHash()
{
  free(NodesInfo);
  free_lvector((unsigned long *) RN, 1, hashSize);
  free_cvector(hashFlag, 1, hashSize);
}


/* -----------------------------------------------------------
  Reads the .aux file to get the other file names
  
  creates extern vars:
     nodesFile[], netsFile[], wtsFile[], sclFile[], 
     plFile[], benchmarkName[];
----------------------------------------------------------- */
void readAuxFile(char benchmarkPath[], char auxFile[]) 
{
    FILE *fp;
    char temp[BUFFERSIZE], placementType[BUFFERSIZE], *name;

  
    strcpy(temp, benchmarkPath);
    strcat(temp, "/");
    strcat(temp, auxFile);
   
    if((fp=fopen(temp, "r")) == NULL) {
        printf("Error in opening: %s \n", auxFile);
        exit(1);
    }
//    printf("Reading %s ...\n",auxFile);
    
    if(fscanf(fp, "%s\t:\t%s%s%s%s%s\n", placementType, nodesFile, netsFile, wtsFile, plFile, sclFile) != 6) abort();

    strcpy(temp, auxFile);
    name = strtok(temp, ".");
    strcpy(benchmarkName, name);

    fclose(fp);
}  


/* -----------------------------------------------------------
  Reads the .nodes file to get cell widths and heights
  
  creates extern vars: 
     movableNodes, numTerminals, averageCellWidth, 
     cellWidth[], cellHeight[]
----------------------------------------------------------- */
void readNodesFile(char benchmarkPath[], char nodesFile[])
{
    FILE *fp;
    char line[BUFFERSIZE], tempStr[BUFFERSIZE], s4[STRINGLEN];
    long j, nodeIndex, nodeNo, currentPos;
    long hashfunc, RN_index;
    float nodeWidth, nodeHeight, sumWidth;

    
    strcpy(tempStr, benchmarkPath);
    strcat(tempStr, "/");
    strcat(tempStr, nodesFile);
    
    if((fp=fopen(tempStr, "r"))==NULL) {
        printf("Error in opening %s file \n", nodesFile);
        exit(1);
    }
//    printf("Reading %s ...\n", nodesFile);

    // Reading first few lines 
    if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    do {
        currentPos = ftell(fp);
        if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  

    if(fscanf(fp, "NumNodes\t:\t%d\n", &numNodes) != 1) abort();
    if(fscanf(fp, "NumTerminals\t:\t%d\n", &numTerminals) != 1) abort();

    do {
       currentPos = ftell(fp);
       if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  
    
    movableNodes = numNodes - numTerminals;       // global var - num of movable cells
    cellWidth = vector(1, numNodes);               // global vector giving cell widths
    cellHeight = vector(1, numNodes);             // global vector giving cell heights
   
    sumWidth = 0;
    
    for(nodeNo=1;nodeNo<=numNodes;nodeNo++) {

        if(!fgets(line, BUFFERSIZE, fp)) abort();
        strcpy(s4, "");
        sscanf(line, "%s%f%f%s\n", tempStr, &nodeWidth, &nodeHeight, s4);

        if(strcmp(s4, "terminal")==0) {

            // find the nodeIndex corresponding to tempStr
            hashfunc = 0;
            for(j=1;j<=IMIN(strlen(tempStr), modresNum);j++)
                hashfunc += ((long)tempStr[j-1]<<3*(j-1))%hashSize;
      
            hashfunc = hashfunc%hashSize;
            RN_index = 1;
  
            while(strcmp(tempStr, NodesInfo[hashfunc].name)!=0 && RN_index<hashSize) {
                hashfunc = (hashfunc+RN[RN_index])%hashSize;
                RN_index++;
            }
      
            if (RN_index>=hashSize) {  
                printf("cannot find in hash table\n");
                exit(1);
            }
      
            nodeIndex = NodesInfo[hashfunc].index;

            // store cellwidth and cellheight corresponding to nodeIndex
            cellWidth[nodeIndex] = nodeWidth;
            cellHeight[nodeIndex] = nodeHeight;
      
        } else {

            // find the nodeIndex corresponding to tempStr
            hashfunc = 0;
            for(j=1;j<=IMIN(strlen(tempStr), modresNum);j++)
                hashfunc += ((long)tempStr[j-1]<<3*(j-1))%hashSize;
      
            hashfunc = hashfunc%hashSize;
            RN_index = 1;
  
            while(strcmp(tempStr, NodesInfo[hashfunc].name)!=0 && RN_index<hashSize) {
                hashfunc = (hashfunc+RN[RN_index])%hashSize;
                RN_index++;
            }
      
            if (RN_index>=hashSize) {  
                printf("cannot find in hash table\n");
                exit(1);
            }
            
            nodeIndex = NodesInfo[hashfunc].index;

            // store cellwidth and cellheight corresponding to nodeIndex
            cellWidth[nodeIndex] = nodeWidth;
            cellHeight[nodeIndex] = nodeHeight;
            sumWidth += nodeWidth;
        }
    }

    // find average cell width
    averageCellWidth = sumWidth/movableNodes;
    averageCellWidth *= 100;
    averageCellWidth = (int)averageCellWidth;
    averageCellWidth /= 100;

    fclose(fp);  

#if(DEBUG)
int i;
for(i=1; i<=movableNodes+numTerminals; i++) {
    printf("%d  %s  %.2f  %.2f\n", i, cellName[i], cellWidth[i], cellHeight[i]);
}

printf("Avg Cell Width:  %.2f \n", averageCellWidth);    
#endif
}  


/* -----------------------------------------------------------
   Reads the .nets file to get the netlist information
   
   creates extern vars: 
      numNets, numPins, 
      xPinOffset[], yPinOffset[], netlist[], netlistIndex[]
----------------------------------------------------------- */
void readNetsFile(char benchmarkPath[], char netsFile[])
{
    FILE *fp;
    long i, j, k, netNo, nodeIndex;
    long currentPos, startPointer, hashfunc, RN_index;
    char tempStr[BUFFERSIZE], nodeName[BUFFERSIZE];
    int degree, prevElements;
    float xOffset, yOffset;


    strcpy(tempStr, benchmarkPath);
    strcat(tempStr, "/");
    strcat(tempStr, netsFile);

    if((fp=fopen(tempStr, "r"))==NULL) {
        printf("Error in opening %s file \n", netsFile);
        exit(1);
    }
//    printf("Reading %s ...\n", netsFile);

    // Reading first four lines 
    if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    do {
        currentPos = ftell(fp);
        if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  

    // getting numNets and numPins
    if(fscanf(fp, "NumNets\t:\t%d\n", &numNets) != 1) abort();
    if(fscanf(fp, "NumPins\t:\t%d\n", &numPins) != 1) abort();

    do {
        currentPos = ftell(fp);
        if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  
   
    // stores the netlist and pin offsets relative to the center of the cells
    netlist = ivector(1,numPins+1);
    xPinOffset = vector(1,numPins+1);
    yPinOffset = vector(1,numPins+1);

    // index vector for the netlist and offset vectors
    netlistIndex = ivector(0,numNets+1);
    
    netlistIndex[0] = 1;
    prevElements = 0;

    for(netNo=1;netNo<=numNets;netNo++) {

        do {
            currentPos = ftell(fp);
            if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
        } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  

        sscanf(tempStr, "NetDegree\t:\t%d\n", &degree);

        netlistIndex[netNo] = netlistIndex[netNo-1] + prevElements;
        startPointer = netlistIndex[netNo];
        prevElements = degree;
      
        for(k=1;k<=degree;k++) {
         
            do {
                currentPos = ftell(fp);
                if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
            } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
        
            xOffset = yOffset = 0.0;
            sscanf(tempStr, "%s%f%f", nodeName, &xOffset, &yOffset);

            // find the nodeIndex corresponding to nodeName
            hashfunc = 0;
            for(j=1;j<=IMIN(strlen(nodeName), modresNum);j++)
                hashfunc += ((long)nodeName[j-1]<<3*(j-1))%hashSize;
        
            hashfunc = hashfunc%hashSize;
            RN_index = 1;
   
            while(strcmp(nodeName, NodesInfo[hashfunc].name)!=0 && RN_index<hashSize) {
                hashfunc = (hashfunc+RN[RN_index])%hashSize;
                RN_index++;
            }
        
            if (RN_index>=hashSize) {  
                printf("cannot find in hash table\n");
               exit(1);
            }

            nodeIndex = NodesInfo[hashfunc].index;
            netlist[startPointer+k-1] = nodeIndex;
            xPinOffset[startPointer+k-1] = xOffset;
            yPinOffset[startPointer+k-1] = yOffset; 
        }
    }
   
    netlistIndex[numNets+1] = netlistIndex[numNets] + prevElements;
    netlist[netlistIndex[numNets+1]] = 0;

    fclose(fp); 

#if(DEBUG)
for(i=1; i<=numNets; i++) {
    printf("**%d**  ", netlistIndex[i+1]-netlistIndex[i]);
    for(j=netlistIndex[i]; j<netlistIndex[i+1]; j++) {
        printf("(%d) %.2f %.2f  ", netlist[j], xPinOffset[j], yPinOffset[j]);
    }
    printf("\n");    
}
#endif
}


/* -----------------------------------------------------------
  Reads the .pl file to get coordinates of all nodes and the 
  placement boundary based on the position of the I/O pads
  
  creates extern vars:
     xCellCoord[], yCellCoord[]
     areaArrayIO[], numAreaArrayIO
----------------------------------------------------------- */
void readPlFile(char benchmarkPath[], char plFile[])
{
    FILE *fp;
    long nodeIndex, currentPos, j, hashfunc, RN_index, nodeNo, movable;
    char tempStr[BUFFERSIZE], nodeName[BUFFERSIZE], fixedType[BUFFERSIZE];
    float xCoord, yCoord;

  
    strcpy(tempStr, benchmarkPath);
    strcat(tempStr, "/");
    strcat(tempStr, plFile);

    if((fp=fopen(tempStr, "r"))==NULL) {
        printf("Error in opening %s file \n", plFile);
        exit(1);
    }
//    printf("Reading %s ...\n", plFile);
  
    // Reading first four lines 
    if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    do {
        currentPos = ftell(fp);
        if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
    fseek(fp, currentPos, SEEK_SET);
  
    xCellCoord = vector(1,numNodes);
    yCellCoord = vector(1,numNodes);
    areaArrayIO = ivector(1, numTerminals);
    numAreaArrayIO = 0;

    movable = numNodes-numTerminals;
    for(nodeNo=1; nodeNo<=numNodes; nodeNo++) {

        if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
        strcpy(fixedType, "");
        sscanf(tempStr, "%s%f%f\t:\t%*s%s\n", nodeName, &xCoord, &yCoord, fixedType);

        hashfunc = 0;
        for(j=1;j<=IMIN(strlen(nodeName), modresNum);j++)
            hashfunc += ((long)nodeName[j-1]<<3*(j-1))%hashSize;
      
        hashfunc = hashfunc%hashSize;
        RN_index = 1;
    
        while(strcmp(nodeName, NodesInfo[hashfunc].name)!=0 && RN_index<hashSize) {
            hashfunc = (hashfunc+RN[RN_index])%hashSize;
            RN_index++;
        }
      
        if (RN_index>=hashSize) {  
            printf("cannot find in hash table\n");
            exit(1);
        }
      
        nodeIndex = NodesInfo[hashfunc].index;
        // Assume all coordinates are integers
        xCellCoord[nodeIndex] = (int) (xCoord + 0.5*cellWidth[nodeIndex]);
        yCellCoord[nodeIndex] = (int) (yCoord + 0.5*cellHeight[nodeIndex]);
/*        
        if(nodeIndex > movable) {
            // Is a fixed terminal but can allow overlap with it
            if(strcmp(fixedType, "/FIXED") != 0) {
                numAreaArrayIO++;
                areaArrayIO[numAreaArrayIO] = nodeIndex;
            }
        }
*/
    }

    fclose(fp);
}    


/* -----------------------------------------------------------
  Reads the .scl file to get placement region information
  
  creates extern vars:
     siteOriginX, siteEndX, siteOriginY, siteEndY, siteWidth, 
     siteSpacing, coreRowHeight, coreWidth, coreHeight, 
     numRows, minX, maxX, minY, maxY
----------------------------------------------------------- */
void readSclFile(char benchmarkPath[], char sclFile[])
{
    FILE *fp;
    char tempStr[BUFFERSIZE], siteOrient[2], siteSymmetry[2], junk[BUFFERSIZE];
    int totalSites, row;
    long currentPos, nodeIndex, movable;
    float originY, originX, minOrigin, maxEnd;


    strcpy(tempStr, benchmarkPath);
    strcat(tempStr, "/");
    strcat(tempStr, sclFile);

    if((fp=fopen(tempStr, "r"))==NULL) {
      printf("Error in opening %s file \n", sclFile);
      exit(1);
    }
//    printf("Reading %s ...\n", sclFile);

    // Reading first four lines 
    if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    do {
      currentPos = ftell(fp);
      if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  
   
    // getting numRows
    if(fscanf(fp, "%*s\t:\t%d\n", &numRows) != 1) abort();

    rowOriginX = vector(1, numRows);
    rowEndX = vector(1, numRows);
    xRowBlockage = vector(1, 2*numRows);
    yRowBlockage = vector(1, 2*numRows);
    widthRowBlockage = vector(1, 2*numRows);

    // any blanks or comments after numRows line
    do {
      currentPos = ftell(fp);
      if(!fgets(tempStr, BUFFERSIZE, fp)) abort();
    } while( (tempStr[0] == '#') || (strlen(tempStr) < 5) );  
    fseek(fp, currentPos, SEEK_SET);  

    siteOriginX = 1.0e6;
    siteEndX = -1.0e6;
    for(row=1; row<=numRows; row++) {
        // Reading CoreRow Horizontal
        if(!fgets(junk, BUFFERSIZE, fp)) abort();
    
        if(fscanf(fp, "\tCoordinate\t:\t%f\n", &originY) != 1) abort();
        if(row == 1) siteOriginY = originY;
        
        if(fscanf(fp, "Height\t:\t%f\n", &coreRowHeight) != 1) abort();
        if(fscanf(fp, "Sitewidth\t:\t%f\n", &siteWidth) != 1) abort();
        if(fscanf(fp, "Sitespacing\t:\t%f\n", &siteSpacing) != 1) abort();
        if(fscanf(fp, "Siteorient\t:\t%s\n", siteOrient) != 1) abort();
        if(fscanf(fp, "Sitesymmetry\t:\t%s\n", siteSymmetry)!= 1) abort();
        if(fscanf(fp, "SubrowOrigin\t:\t%f\t%*s\t:\t%d\n", &originX, &totalSites) != 2) abort();
        // Reading End
        if(!fgets(junk, BUFFERSIZE, fp)) abort();
      
        rowOriginX[row] = originX;
        rowEndX[row] = originX + totalSites*siteSpacing;
        if(rowOriginX[row] < siteOriginX) siteOriginX = rowOriginX[row];
        if(rowEndX[row] > siteEndX) siteEndX = rowEndX[row];
    }

    siteEndY = numRows*coreRowHeight + siteOriginY;
    coreHeight = siteEndY - siteOriginY;          // height of placement area 
    coreWidth = siteEndX - siteOriginX;

    numRowBlockages = 0;
    for(row=1; row<=numRows; row++) {
        if(rowOriginX[row] > siteOriginX) {
            numRowBlockages++;
            xRowBlockage[numRowBlockages] = siteOriginX + 0.5*(rowOriginX[row] - siteOriginX);
            yRowBlockage[numRowBlockages] = siteOriginY + (row-0.5)*coreRowHeight;
            widthRowBlockage[numRowBlockages] = (rowOriginX[row] - siteOriginX);
        }

        if(siteEndX > rowEndX[row]) {
            numRowBlockages++;
            xRowBlockage[numRowBlockages] = rowEndX[row] + 0.5*(siteEndX - rowEndX[row]);
            yRowBlockage[numRowBlockages] = siteOriginY + (row-0.5)*coreRowHeight;
            widthRowBlockage[numRowBlockages] = (siteEndX - rowEndX[row]);
        }
    }
    
    maxX = 0;
    minX = 1.0e10;
    maxY = 0;
    minY = 1.0e10;
    movable = numNodes-numTerminals;
    for(nodeIndex=movable+1; nodeIndex<=numNodes; nodeIndex++) {
    
        if(xCellCoord[nodeIndex] > maxX) maxX = xCellCoord[nodeIndex];
        if(xCellCoord[nodeIndex] < minX) minX = xCellCoord[nodeIndex];
        if(yCellCoord[nodeIndex] > maxY) maxY = yCellCoord[nodeIndex];
        if(yCellCoord[nodeIndex] < minY) minY = yCellCoord[nodeIndex];
    }
    maxX = MAX(maxX, siteEndX+5.0);
    minX = MIN(minX, siteOriginX-5.0);
    maxY = MAX(maxY, siteEndY+5.0);
    minY = MIN(minY, siteOriginY-5.0);   
    
    fclose(fp);
}  


/* -----------------------------------------------------------
   writes out a bookshelf format .pl file
----------------------------------------------------------- */
void writePlFile(char outputDir[], char benchmarkName[], float xCoord[], float yCoord[]) 
{
    FILE *fp;
    char tempStr[BUFFERSIZE];
    int i;


    strcpy(tempStr, outputDir);
    strcat(tempStr, "/");
    strcat(tempStr, benchmarkName);
    strcat(tempStr, "_out.pl");
    
    if( (fp=fopen(tempStr,"w")) == NULL ) {
     
        printf("ERROR in opening the %s file for write \n", tempStr);
        exit(1);
    }
    printf("\nPrinting %s File... \n",tempStr);

    fprintf(fp, "UCLA pl 1.0\n");
    fprintf(fp, "# Circuit  :  %s\n", benchmarkName);

    for(i=1; i<=numNodes; i++)
        fprintf(fp, "    %20s    %-10.2f    %-10.2f  :  N\n", 
                cellName[i], xCoord[i]-0.5*cellWidth[i], yCoord[i]-0.5*cellHeight[i]);
   
    fclose(fp);
}
