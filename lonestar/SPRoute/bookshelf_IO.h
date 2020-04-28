/* -----------  FastPlace - Version 1.0 ----------------
                       by 
   Natarajan Viswanathan and Chris C.-N. Chu
     Dept. of ECpE, Iowa State University
          Copyright (c) - 2004 
Iowa State University Research Foundation, Inc.
-------------------------------------------------------- */
/* --------------------------------------------------------------------------
   Header file used in bookshelf_IO.c 
----------------------------------------------------------------------------*/

#ifndef _BOOKSHELF_IO_H_
#define _BOOKSHELF_IO_H_

#define BUFFERSIZE 200
#define STRINGLEN 30

    /* -----------------------------------------------------------------------------
        Reads the .nodes file and creates a hash linking cell name to cell index for 
        all the nodes in the circuit (movable nodes + fixed nodes + I/O pads)

        creates extern vars:
            cellName[i]     (i = 1..movableNodes + numTerminals)
    -------------------------------------------------------------------------------- */
    extern void createHash(char benchmarkPath[], char nodesFile[]);
    extern void freeHash();

    /* -----------------------------------------------------------------------------
        Reads the .aux file to get the other file names
  
        creates extern vars:
            nodesFile[], netsFile[], wtsFile[], sclFile[], plFile[], benchmarkName[]
    -------------------------------------------------------------------------------- */
    extern void readAuxFile(char benchmarkPath[], char auxFile[]);

    /* -----------------------------------------------------------------------------
        Reads the .nodes file to get cell widths and heights
        
        creates extern vars: 
            movableNodes, numTerminals, averageCellWidth, cellWidth[], cellHeight[]
    -------------------------------------------------------------------------------- */
    extern void readNodesFile(char benchmarkPath[], char nodesFile[]);

    /* -----------------------------------------------------------------------------
        Reads the .nets file to get the netlist information
   
        creates extern vars: 
            numNets, numPins, netlist[], netlistIndex[], xPinOffset[], yPinOffset[]
    -------------------------------------------------------------------------------- */
    extern void readNetsFile(char benchmarkPath[], char netsFile[]);

    /* -----------------------------------------------------------------------------
        Reads the .pl file to get coordinates of all the nodes and the placement 
        boundary based on the position of the I/O pads
  
        creates extern vars:
            xCellCoord[], yCellCoord[], minX, maxX, minY, maxY
    -------------------------------------------------------------------------------- */
    extern void readPlFile(char benchmarkPath[], char plFile[]);
    
    /* -----------------------------------------------------------------------------
        Reads the .scl file to get placement (core) region information
  
        creates extern vars:
            siteOriginX, siteEndX, siteOriginY, siteEndY, siteWidth, siteSpacing, 
            numRows, coreRowHeight, coreWidth, coreHeight 
    -------------------------------------------------------------------------------- */
    extern void readSclFile(char benchmarkPath[], char sclFile[]);
    
    /* -----------------------------------------------------------------------------
        writes out a bookshelf format .pl file
    -------------------------------------------------------------------------------- */
    extern void writePlFile(char outputDir[], char benchmarkName[], float xCoord[], float yCoord[]);


    /*--------------  Extern Variables  ------------------*/

    extern char **cellName;
    
    extern char nodesFile[BUFFERSIZE], netsFile[BUFFERSIZE], wtsFile[BUFFERSIZE];
    extern char sclFile[BUFFERSIZE], plFile[BUFFERSIZE], benchmarkName[BUFFERSIZE];

    extern int movableNodes, numTerminals;
    extern float averageCellWidth, *cellWidth, *cellHeight; 
    
    extern int numNets, numPins, *netlist, *netlistIndex;
    extern float *xPinOffset, *yPinOffset;
       
    extern float *xCellCoord, *yCellCoord, minX, maxX, minY, maxY;
    extern int *areaArrayIO, numAreaArrayIO;

    extern int numRows, numRowBlockages;
    extern float siteOriginY, siteEndY, coreHeight;
    extern float siteOriginX, siteEndX, coreWidth;
    extern float siteWidth, siteSpacing, coreRowHeight;
    extern float *rowOriginX, *rowEndX;
    extern float *xRowBlockage, *yRowBlockage, *widthRowBlockage;


#endif /* _BOOKSHELF_IO_H_*/ 

/* -----------------------------------------------------------------------------------------------
                                Description of Extern Variables

    cellName[i]         =   cell name corresponding to cell index "i" 
                            (i = 1..movableNodes+numTerminals)

    movableNodes        =   number of movable nodes,
    numTerminals        =   number of fixed nodes (I/O Pads + Fixed Macros)
    averageCellWidth    =   avg width of movable nodes,
    cellWidth[i]        =   width of cell "i"   (i = 1..movableNodes+numTerminals)
    cellHeight[i]       =   height of cell "i"  (i = 1..movableNodes+numTerminals)

    numNets             =   number of nets
    numPins             =   number of pins
    netlist[]           =   netlist of the circuit
    xPinOffset[]        =   x-offset of the pins from the center of the cell
    yPinOffset[]        =   y-offset (      "       )
    netlistIndex[]      =   index to the netlist and offset vectors

    xCellCoord[i]       =   x-coordinate of cell "i"  (i = 1..movableNodes+numTerminals)
    yCellCoord[i]       =   y-coordinate of cell "i", 
    minX, maxX          =   left and right boundaries of the chip (Note: not the placement region) 
    minY, maxY          =   bottom and top boundaries of the chip
    areaArrayIO[]       =   All fixed terminals with which there can be an overlap of movable nodes
                            (eg area array IOs)
    numAreaArrayIO      =   the total number of area array IOs

    siteOriginX[]       =   left boundary of the placement region within a row
    siteEndX[]          =   right boundary of the placement region within a row   
    siteOriginY         =   bottom boundary of the placement region    
    siteEndY            =   top boundary of the placement region   
    siteWidth           =   width of a placement site within a row 
    siteSpacing         =   the space b/w adjacent placement sites within a row 
    numRows             =   number of placement rows 
    coreRowHeight       =   row Height 
    coreWidth           =   siteEndX - siteOriginX 
    coreHeight          =   siteEndY - siteOriginY 

---------------------------------------------------------------------------------------------------*/
