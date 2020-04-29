#ifndef _MAZE3D_H_
#define _MAZE3D_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "DataType.h"
#include "flute.h"
#include "DataProc.h"
#include "route.h"
#include "RipUp.h"
#include <time.h>

#define PARENT(i) (i-1)/2
//#define PARENT(i) ((i-1)>>1)
#define LEFT(i) 2*i+1
#define RIGHT(i) 2*i+2

typedef struct
{
    int x;   // x position
    int y;   // y position
	int l;
} Pos3D;
// non recursive version of heapify-
static void heapify3D(int **array, int heapSize, int i)
{
    int l, r, smallest;
    int *tmp;
    Bool STOP = FALSE;
    
    tmp = array[i];
    do{
    
        l = LEFT(i);
        r = RIGHT(i);

        if(l<heapSize && *(array[l])<*tmp)
        {
            smallest = l;
            if(r<heapSize && *(array[r])<*(array[l]))
				smallest = r;
        }
        else
        {
            smallest = i;
            if(r<heapSize && *(array[r])<*tmp)
				smallest = r;
        }
        if(smallest!=i)
        {
            array[i] = array[smallest];
            i = smallest;
        }
        else 
        {
            array[i] = tmp;
            STOP = TRUE;
        }
    }
    while (!STOP);
}

// build heap for an list of grid
static void buildHeap3D(int **array, int arrayLen)
{
    int i;
    
    for (i=arrayLen/2-1; i>=0; i--)
        heapify3D(array, arrayLen, i);
}

static void updateHeap3D(int **array, int arrayLen, int i)
{


    int parent;
    int *tmp, *tmpi;
		
//	printf("heap being updated");
//	fflush(stdout);
    
    tmpi = array[i];
    while(i>0 && *(array[PARENT(i)])>*tmpi)
    {
        parent = PARENT(i);
        array[i] = array[parent];
        i = parent;
    }
    array[i] = tmpi;

//	printf("heap updated succedded");
//	fflush(stdout);
    
}

// extract the entry with minimum distance from Priority queue
static void extractMin3D(int **array, int arrayLen)
{
    
    if(arrayLen<1)
        printf("Error: heap underflow\n");
    array[0] = array[arrayLen-1];
    heapify3D(array, arrayLen-1, 0);
}




void setupHeap3D(int netID, int edgeID, int* heapLen1, int* heapLen2, int regionX1, int regionX2, int regionY1, int regionY2)
{
    int i, j, k,l,d, numNodes, x1, y1, x2, y2, max_y, min_y, layer, topL, botL, n1, n2, nt, n1a, n2a;
    int nbr, nbrX, nbrY, nbrL, cur, edge, edge_n1, edge_n2, edge_x1, edge_y1, edge_x2, edge_y2;
    int grid, x_grid, y_grid,l_grid, heapcnt;
    int queuehead, queuetail, routeLen;
    int *heapQueue;
 //   Bool *heapVisited;
    TreeEdge *treeedges;
    TreeNode *treenodes;
    Route *route;
    //Bool **inRegion;  // the flag to check weather the node is in the enlarged region for maze routing

    //inRegion = (Bool**) calloc(yGrid, sizeof(Bool*));
    //for(i=0; i<yGrid; i++)
        //inRegion[i] = (Bool*) calloc(xGrid, sizeof(Bool));
    
    
    
    
/*	for ( l = 0; l < numLayers; l++) {
		for(i=regionY1; i<=regionY2; i++) {
			for(j=regionX1; j<=regionX2; j++) {
				corrEdge3D[l][i][j] = BIG_INT;
			}
		}
	}*/

	//return;
    
    treeedges = sttrees[netID].edges;
    treenodes = sttrees[netID].nodes;
    d = sttrees[netID].deg;
	routeLen = treeedges[edgeID].route.routelen;

    n1 = treeedges[edgeID].n1;
    n2 = treeedges[edgeID].n2;
	n1a = treeedges[edgeID].n1a;
	n2a = treeedges[edgeID].n2a;
    x1 = treenodes[n1].x;
    y1 = treenodes[n1].y;
    x2 = treenodes[n2].x;
    y2 = treenodes[n2].y;


    if(d==2) // 2-pin net
    {
//		printf("2pinnet l1 %d, l2 %d\n", l1, l2);
        d13D[0][y1][x1] = 0;
		directions3D[0][y1][x1] = ORIGIN;
        heap13D[0] = &(d13D[0][y1][x1]);
        *heapLen1 = 1;
        d23D[0][y2][x2] = 0;
		directions3D[0][y2][x2] = ORIGIN;
        heap23D[0] = &(d23D[0][y2][x2]);
        *heapLen2 = 1;
    }
    else // net with more than 2 pins
    {
    	heapQueue = (int*) calloc(MAXNETDEG, sizeof(int));

		for(i=regionY1; i<=regionY2; i++) {
			for(j=regionX1; j<=regionX2; j++) {
				inRegion[i][j] = TRUE;
			}
		}
		
        numNodes = 2*d-2;

		
        for(i=0; i<numNodes; i++)
            heapVisited[i] = FALSE;


        // find all the grids on tree edges in subtree t1 (connecting to n1) and put them into heap13D
        if(n1<d) // n1 is a Pin node
        {
//			getLayerRange(treenodes, treeedges ,n1, edgeID, &topL, &botL);

				// just need to put n1 itself into heap13D
			heapcnt = 0;


			nt =  treenodes[n1].stackAlias;

			for (l = treenodes[nt].botL; l <=treenodes[nt].topL; l++) {
				d13D[l][y1][x1] = 0;
				//cout << heap13D << " " << heapcnt << " " << d13D[l][y1][x1] << " " << &(d13D[l][y1][x1]) << endl;
				heap13D[heapcnt] = &(d13D[l][y1][x1]);
				directions3D[l][y1][x1] = ORIGIN;
				heapVisited[n1] = TRUE;
				heapcnt ++;
			}
			*heapLen1 = heapcnt;

        }
        else // n1 is a Steiner node
        {
            heapcnt = 0;
            queuehead = queuetail = 0;

//			getLayerRange(treenodes, treeedges ,n1, edgeID, &topL, &botL);

			nt =  treenodes[n1].stackAlias;

				// add n1 into heap13D
			for (l = treenodes[nt].botL; l <=treenodes[nt].topL; l++) {
				d13D[l][y1][x1] = 0;
				directions3D[l][y1][x1] = ORIGIN;
				heap13D[heapcnt] = &(d13D[l][y1][x1]);
				heapVisited[n1] = TRUE;
				heapcnt++;
			}

            // add n1 into the heapQueue
            heapQueue[queuetail] = n1;
            queuetail++;

            // loop to find all the edges in subtree t1
            while(queuetail>queuehead)
            {
// get cur node from the queuehead
				cur = heapQueue[queuehead];
				queuehead++;
				heapVisited[cur] = TRUE;
				if(cur>=d) // cur node is a Steiner node
				{
					for(i=0; i<3; i++)
					{
						nbr = treenodes[cur].nbr[i];
						edge = treenodes[cur].edge[i];
						if(nbr!=n2) // not n2
						{
							if(heapVisited[nbr]==FALSE)
							{
								
								// put all the grids on the two adjacent tree edges into heap13D
								if(treeedges[edge].route.routelen>0) // not a degraded edge
								{
								
									
									// put nbr into heap13D if in enlarged region
									if(inRegion[treenodes[nbr].y][treenodes[nbr].x])
									{
										nbrX = treenodes[nbr].x;
										nbrY = treenodes[nbr].y;
										nt =  treenodes[nbr].stackAlias;
										for (l = treenodes[nt].botL; l <=treenodes[nt].topL; l++) {
											
											d13D[l][nbrY][nbrX] = 0;
											directions3D[l][nbrY][nbrX] = ORIGIN;
											heap13D[heapcnt] = &(d13D[l][nbrY][nbrX]);
											heapcnt ++;
											corrEdge3D[l][nbrY][nbrX] = edge;
										}
									}

									// the coordinates of two end nodes of the edge

									
									route = &(treeedges[edge].route);
									if(route->type==MAZEROUTE)
									{
										for(j=1; j<route->routelen; j++) // don't put edge_n1 and edge_n2 into heap13D
										{
											x_grid = route->gridsX[j];
											y_grid = route->gridsY[j];
											l_grid = route->gridsL[j];

											if(inRegion[y_grid][x_grid])
											{
												d13D[l_grid][y_grid][x_grid] = 0;
												heap13D[heapcnt] = &(d13D[l_grid][y_grid][x_grid]);
												directions3D[l_grid][y_grid][x_grid] = ORIGIN;
												heapcnt ++;
												corrEdge3D[l_grid][y_grid][x_grid] = edge;

											}
										}

									} // if MAZEROUTE
								} // if not a degraded edge (len>0)

								// add the neighbor of cur node into heapQueue
								heapQueue[queuetail] = nbr;
								queuetail++;
							} // if the node is not heapVisited
						} // if nbr!=n2
					} // loop i (3 neigbors for cur node)
				} // if cur node is a Steiner nodes
            } // while heapQueue is not empty
            *heapLen1 = heapcnt; // record the length of heap13D
        } // else n1 is not a Pin node

        // find all the grids on subtree t2 (connect to n2) and put them into heap23D
        // find all the grids on tree edges in subtree t2 (connecting to n2) and put them into heap23D
        if(n2<d) // n2 is a Pin node
        {

			nt =  treenodes[n2].stackAlias;
			//*heapLen2 = 0;
			heapcnt = 0;

			for (l = treenodes[nt].botL; l <=treenodes[nt].topL; l++) {
            // just need to put n1 itself into heap13D
				d23D[l][y2][x2] = 0;
				directions3D[l][y2][x2] = ORIGIN;
				heap23D[heapcnt] = &(d23D[l][y2][x2]);
				heapVisited[n2] = TRUE;
				//*heapLen2 += 1;
				heapcnt ++;
			}
			*heapLen2 = heapcnt;
        }
        else // n2 is a Steiner node
        {
            heapcnt = 0;
            queuehead = queuetail = 0;

			nt =  treenodes[n2].stackAlias;
            // add n2 into heap23D
			for (l = treenodes[nt].botL; l <=treenodes[nt].topL; l++) {
				d23D[l][y2][x2] = 0;
				directions3D[l][y2][x2] = ORIGIN;
				heap23D[heapcnt] = &(d23D[l][y2][x2]);
				heapcnt++;
			}
			heapVisited[n2] = TRUE;

            // add n2 into the heapQueue
            heapQueue[queuetail] = n2;
            queuetail++;

            // loop to find all the edges in subtree t2
            while(queuetail>queuehead)
            {
				// get cur node form queuehead
				cur = heapQueue[queuehead];
				heapVisited[cur] = TRUE;
				queuehead++;

				if(cur>=d) // cur node is a Steiner node
				{
					for(i=0; i<3; i++)
					{
						nbr = treenodes[cur].nbr[i];
						edge = treenodes[cur].edge[i];
						if(nbr!=n1) // not n1
						{
							if(heapVisited[nbr]==FALSE)
							{
								// put all the grids on the two adjacent tree edges into heap23D
								if(treeedges[edge].route.routelen>0) // not a degraded edge
								{

									
									// put nbr into heap23D
									if(inRegion[treenodes[nbr].y][treenodes[nbr].x])
									{
										nbrX = treenodes[nbr].x;
										nbrY = treenodes[nbr].y;
										nt =  treenodes[nbr].stackAlias;
										for (l = treenodes[nt].botL; l <=treenodes[nt].topL; l++) {
											//nbrL = treenodes[nbr].l;
											
											d23D[l][nbrY][nbrX] = 0;
											directions3D[l][nbrY][nbrX] = ORIGIN;
											heap23D[heapcnt] = &(d23D[l][nbrY][nbrX]);
											heapcnt ++;
											corrEdge3D[l][nbrY][nbrX] = edge;
										}
									}

									// the coordinates of two end nodes of the edge
									
									
									route = &(treeedges[edge].route);
									if(route->type==MAZEROUTE)
									{
										
										for(j=1; j<route->routelen; j++) // don't put edge_n1 and edge_n2 into heap23D
										{
											x_grid = route->gridsX[j];
											y_grid = route->gridsY[j];
											l_grid = route->gridsL[j];
											if(inRegion[y_grid][x_grid])
											{
												
												d23D[l_grid][y_grid][x_grid] = 0;
												directions3D[l_grid][y_grid][x_grid] = ORIGIN;
												heap23D[heapcnt] = &(d23D[l_grid][y_grid][x_grid]);
												heapcnt ++;

												corrEdge3D[l_grid][y_grid][x_grid] = edge;
											}
										}
										
									} // if MAZEROUTE
								} // if the edge is not degraded (len>0)

								// add the neighbor of cur node into heapQueue
								heapQueue[queuetail] = nbr;
								queuetail++;
							} // if the node is not heapVisited
						} // if nbr!=n1
					} // loop i (3 neigbors for cur node)
				} // if cur node is a Steiner nodes
            } // while heapQueue is not empty
            *heapLen2 = heapcnt; // record the length of heap23D
        } // else n2 is not a Pin node

	//	printf("queuetail %d, numnodes %d\n", queuetail, numNodes);
	//	fflush(stdout);
        free(heapQueue);
     //   free(heapVisited);

	//	printf("there after\n", queuetail, numNodes);
	//	fflush(stdout);
		
		for(i=regionY1; i<=regionY2; i++) {
			for(j=regionX1; j<=regionX2; j++) {
				inRegion[i][j] = FALSE;
			}
		}
	} // net with more than two pins


	//for(i=0; i<yGrid; i++) {
        //free(inRegion[i]);
	//}
    //free(inRegion);
}



void newUpdateNodeLayers(TreeNode *treenodes, int edgeID, int n1, int lastL) 
{
	int i, con;

	con = treenodes[n1].conCNT;

	treenodes[n1].heights[con] = lastL;
	treenodes[n1].eID[con] = edgeID;
	treenodes[n1].conCNT ++;
	if (treenodes[n1].topL < lastL) {
		treenodes[n1].topL = lastL;
		treenodes[n1].hID = edgeID;
	} 
	if (treenodes[n1].botL > lastL) {
		treenodes[n1].botL = lastL;
		treenodes[n1].lID = edgeID;
	}
}




int copyGrids3D(TreeNode *treenodes, int n1, int n2, TreeEdge *treeedges, int edge_n1n2, int gridsX_n1n2[], int gridsY_n1n2[], int gridsL_n1n2[])
{
    int i, cnt;
    int n1x, n1y, n2x, n2y, Zpoint, n1l, n2l;
    
    n1x = treenodes[n1].x;
    n1y = treenodes[n1].y;
    n2x = treenodes[n2].x;
    n2y = treenodes[n2].y;
	//n1l = treenodes[n1].l;
	//n2l = treenodes[n2].l;


    cnt = 0;
    if(treeedges[edge_n1n2].n1==n1) // n1 is the first node of (n1, n2)
    {
        if(treeedges[edge_n1n2].route.routelen > 0)
        {
            for(i=0; i<=treeedges[edge_n1n2].route.routelen; i++)
            {
				gridsX_n1n2[cnt] = treeedges[edge_n1n2].route.gridsX[i];
				gridsY_n1n2[cnt] = treeedges[edge_n1n2].route.gridsY[i];
				gridsL_n1n2[cnt] = treeedges[edge_n1n2].route.gridsL[i];
				cnt++;
            }
        } // MAZEROUTE
        else // NOROUTE
        {
			fflush(stdout);
            gridsX_n1n2[cnt] = n1x;
            gridsY_n1n2[cnt] = n1y;
			gridsL_n1n2[cnt] = n1l;
            cnt++;
        }
    } // if n1 is the first node of (n1, n2)
    else  // n2 is the first node of (n1, n2)
    {
        if(treeedges[edge_n1n2].route.routelen > 0)
        {
            for(i=treeedges[edge_n1n2].route.routelen; i>=0; i--)
            {
				gridsX_n1n2[cnt] = treeedges[edge_n1n2].route.gridsX[i];
				gridsY_n1n2[cnt] = treeedges[edge_n1n2].route.gridsY[i];
				gridsL_n1n2[cnt] = treeedges[edge_n1n2].route.gridsL[i];
				cnt++;
            }
        } // MAZEROUTE
        else // NOROUTE
        {
            gridsX_n1n2[cnt] = n1x;
            gridsY_n1n2[cnt] = n1y;
			gridsL_n1n2[cnt] = n1l;
            cnt++;
        } // MAZEROUTE
    }

    return(cnt);
}




void updateRouteType13D(int netID, TreeNode *treenodes, int n1, int A1, int A2, int E1x, int E1y, TreeEdge *treeedges, int edge_n1A1, int edge_n1A2)
{
    int i,l, cnt, A1x, A1y,A1l, A2x, A2y, A2l,Zpoint;
    int cnt_n1A1, cnt_n1A2, E1_pos1, E1_pos2, pos1L, pos2L, nodeADJ;
    int gridsX_n1A1[MAXLEN], gridsY_n1A1[MAXLEN], gridsL_n1A1[MAXLEN],gridsX_n1A2[MAXLEN], gridsY_n1A2[MAXLEN],gridsL_n1A2[MAXLEN];
    
    A1x = treenodes[A1].x;
    A1y = treenodes[A1].y;
    A2x = treenodes[A2].x;
    A2y = treenodes[A2].y;
    
    // copy all the grids on (n1, A1) and (n2, A2) to tmp arrays, and keep the grids order A1->n1->A2
    // copy (n1, A1)
    cnt_n1A1 = copyGrids3D(treenodes, A1, n1, treeedges, edge_n1A1, gridsX_n1A1, gridsY_n1A1, gridsL_n1A1);

    // copy (n1, A2)
    cnt_n1A2 = copyGrids3D(treenodes, n1, A2, treeedges, edge_n1A2, gridsX_n1A2, gridsY_n1A2, gridsL_n1A2);

	if (cnt_n1A1  == 1) {
		printf("in 3D maze routing, type 1 node shift, cnt_n1A1 is 1\n");
		exit(0);
	}

    for(i=0; i<cnt_n1A1; i++)
    {
        if(gridsX_n1A1[i]==E1x && gridsY_n1A1[i]==E1y) // reach the E1
        {
            E1_pos1 = i;
			pos1L = gridsL_n1A1[i];
            break;
        }
    }

	for(i=cnt_n1A1-1; i>=0; i--)
    {
        if(gridsX_n1A1[i]==E1x && gridsY_n1A1[i]==E1y) // reach the E1
        {
            E1_pos2 = i;
			pos2L = gridsL_n1A1[i];
            break;
        }
    }




	
    // reallocate memory for route.gridsX and route.gridsY
    if(treeedges[edge_n1A1].route.type==MAZEROUTE&& treeedges[edge_n1A1].route.routelen > 0) // if originally allocated, free them first
    {
        free(treeedges[edge_n1A1].route.gridsX);
        free(treeedges[edge_n1A1].route.gridsY);
		free(treeedges[edge_n1A1].route.gridsL);
    }
    treeedges[edge_n1A1].route.gridsX = (short*)calloc((E1_pos1+1), sizeof(short));
    treeedges[edge_n1A1].route.gridsY = (short*)calloc((E1_pos1+1), sizeof(short));
	treeedges[edge_n1A1].route.gridsL = (short*)calloc((E1_pos1+1), sizeof(short));


    if(A1x<=E1x)
    {
        cnt = 0;
        for(i=0; i<=E1_pos1; i++)
        {
            treeedges[edge_n1A1].route.gridsX[cnt] = gridsX_n1A1[i];
            treeedges[edge_n1A1].route.gridsY[cnt] = gridsY_n1A1[i];
			treeedges[edge_n1A1].route.gridsL[cnt] = gridsL_n1A1[i];
            cnt++;
        }
        treeedges[edge_n1A1].n1 = A1;
        treeedges[edge_n1A1].n2 = n1;
    }
    else
    {
        cnt = 0;
        for(i=E1_pos1; i>=0; i--)
        {
            treeedges[edge_n1A1].route.gridsX[cnt] = gridsX_n1A1[i];
            treeedges[edge_n1A1].route.gridsY[cnt] = gridsY_n1A1[i];
			treeedges[edge_n1A1].route.gridsL[cnt] = gridsL_n1A1[i];
            cnt++;
        }
        treeedges[edge_n1A1].n1 = n1;
        treeedges[edge_n1A1].n2 = A1;
    }
	treeedges[edge_n1A1].len = ADIFF(A1x,E1x) + ADIFF(A1y,E1y);

    treeedges[edge_n1A1].route.type = MAZEROUTE;
    treeedges[edge_n1A1].route.routelen = E1_pos1;

    // reallocate memory for route.gridsX and route.gridsY
    if(treeedges[edge_n1A2].route.type==MAZEROUTE && treeedges[edge_n1A2].route.routelen > 0) // if originally allocated, free them first
    {
        free(treeedges[edge_n1A2].route.gridsX);
        free(treeedges[edge_n1A2].route.gridsY);
		free(treeedges[edge_n1A2].route.gridsL);
    }

	if (cnt_n1A2 > 1) {
		treeedges[edge_n1A2].route.gridsX = (short*)calloc((cnt_n1A1+cnt_n1A2-E1_pos2-1 + ADIFF(gridsL_n1A1[cnt_n1A1-1], gridsL_n1A2[0])), sizeof(short));
		treeedges[edge_n1A2].route.gridsY = (short*)calloc((cnt_n1A1+cnt_n1A2-E1_pos2-1+ ADIFF(gridsL_n1A1[cnt_n1A1-1], gridsL_n1A2[0])), sizeof(short));
		treeedges[edge_n1A2].route.gridsL = (short*)calloc((cnt_n1A1+cnt_n1A2-E1_pos2-1+ ADIFF(gridsL_n1A1[cnt_n1A1-1], gridsL_n1A2[0])), sizeof(short));
	} else {
		treeedges[edge_n1A2].route.gridsX = (short*)calloc((cnt_n1A1+cnt_n1A2-E1_pos2-1), sizeof(short));
		treeedges[edge_n1A2].route.gridsY = (short*)calloc((cnt_n1A1+cnt_n1A2-E1_pos2-1), sizeof(short));
		treeedges[edge_n1A2].route.gridsL = (short*)calloc((cnt_n1A1+cnt_n1A2-E1_pos2-1), sizeof(short));
	}

    if(E1x<=A2x)
    {
        cnt = 0;
        for(i=E1_pos2; i<cnt_n1A1; i++)
        {
            treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A1[i];
            treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A1[i];
			treeedges[edge_n1A2].route.gridsL[cnt] = gridsL_n1A1[i];
            cnt++;
        }
		if (cnt_n1A2 > 1) {
			if (gridsL_n1A1[cnt_n1A1-1] > gridsL_n1A2[0]) {
				for (l = gridsL_n1A1[cnt_n1A1-1]-1; l >= gridsL_n1A2[0]; l--) {
					treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[0];
					treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[0];
					treeedges[edge_n1A2].route.gridsL[cnt] = l;
					cnt++;
				}
			} else if (gridsL_n1A1[cnt_n1A1-1] < gridsL_n1A2[0]) {
				for (l = gridsL_n1A1[cnt_n1A1-1]+1; l <= gridsL_n1A2[0]; l++) {
					treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[0];
					treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[0];
					treeedges[edge_n1A2].route.gridsL[cnt] = l;
					cnt++;
				}
			}
		}

        for(i=1; i<cnt_n1A2; i++) // 0 is n1 again, so no repeat
        {
            treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[i];
            treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[i];
			treeedges[edge_n1A2].route.gridsL[cnt] = gridsL_n1A2[i];
            cnt++;
        }
        treeedges[edge_n1A2].n1 = n1;
        treeedges[edge_n1A2].n2 = A2;
		A2l = treeedges[edge_n1A2].route.gridsL[cnt-1];
    }
    else
    {
        cnt = 0;
        for(i=cnt_n1A2-1; i>=1; i--) // 0 is n1 again, so no repeat
        {
            treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[i];
            treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[i];
			treeedges[edge_n1A2].route.gridsL[cnt] = gridsL_n1A2[i];
            cnt++;
        }

		if (cnt_n1A2 > 1) {
			if (gridsL_n1A1[cnt_n1A1-1] > gridsL_n1A2[0]) {
				for (l = gridsL_n1A2[0]; l < gridsL_n1A1[cnt_n1A1-1];  l++) {
					treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[0];
					treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[0];
					treeedges[edge_n1A2].route.gridsL[cnt] = l;
					cnt++;
				}
			} else if (gridsL_n1A1[cnt_n1A1-1] < gridsL_n1A2[0]) {
				for (l = gridsL_n1A2[0]; l > gridsL_n1A1[cnt_n1A1-1]; l--) {
					treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[0];
					treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[0];
					treeedges[edge_n1A2].route.gridsL[cnt] = l;
					cnt++;
				}
			}
		}
        for(i=cnt_n1A1-1; i>=E1_pos2; i--)
        {
            treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A1[i];
            treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A1[i];
			treeedges[edge_n1A2].route.gridsL[cnt] = gridsL_n1A1[i];
            cnt++;
        }
        treeedges[edge_n1A2].n1 = A2;
        treeedges[edge_n1A2].n2 = n1;
		A2l = treeedges[edge_n1A2].route.gridsL[0];
    }
    treeedges[edge_n1A2].route.type = MAZEROUTE;
    treeedges[edge_n1A2].route.routelen = cnt-1;
	treeedges[edge_n1A2].len = ADIFF(A2x,E1x) + ADIFF(A2y,E1y);


	treenodes[n1].x = E1x;
	treenodes[n1].y = E1y;

}

void updateRouteType23D(int netID, TreeNode *treenodes, int n1, int A1, int A2, int C1, int C2, int E1x, int E1y, TreeEdge *treeedges, int edge_n1A1, int edge_n1A2, int edge_C1C2)
{
    int i, cnt, A1x, A1y,A1l, A2x, A2y,A2l, C1x, C1y,C1l, C2x, C2y,C2l, Zpoint, extraLen, startIND;
    int edge_n1C1, edge_n1C2, edge_A1A2;
    int cnt_n1A1, cnt_n1A2, cnt_C1C2, E1_pos1, E1_pos2;
    int len_A1A2, len_n1C1, len_n1C2, distance;
    int gridsX_n1A1[MAXLEN], gridsY_n1A1[MAXLEN],gridsL_n1A1[MAXLEN];
    int gridsX_n1A2[MAXLEN], gridsY_n1A2[MAXLEN],gridsL_n1A2[MAXLEN];
    int gridsX_C1C2[MAXLEN], gridsY_C1C2[MAXLEN],gridsL_C1C2[MAXLEN];
    int gridsX_n1C1[MAXLEN], gridsY_n1C1[MAXLEN],gridsL_n1C1[MAXLEN];
    int gridsX_n1C2[MAXLEN], gridsY_n1C2[MAXLEN],gridsL_n1C2[MAXLEN];

    A1x = treenodes[A1].x;
    A1y = treenodes[A1].y;
	//A1l = treenodes[A1].l;
    A2x = treenodes[A2].x;
    A2y = treenodes[A2].y;
    C1x = treenodes[C1].x;
    C1y = treenodes[C1].y;
    C2x = treenodes[C2].x;
    C2y = treenodes[C2].y;
    
	//printf("orig edge_n1A1 %d edge_n1A2 %d edge_C1C2 %d\n",edge_n1A1,edge_n1A2,edge_C1C2 );
    edge_n1C1 = edge_n1A1;
    edge_n1C2 = edge_n1A2;
    edge_A1A2 = edge_C1C2;

    
    // combine (n1, A1) and (n1, A2) into (A1, A2), A1 is the first node and A2 is the second
    // grids order A1->n1->A2
    // copy (A1, n1)
    cnt_n1A1 = copyGrids3D(treenodes, A1, n1, treeedges, edge_n1A1, gridsX_n1A1, gridsY_n1A1,gridsL_n1A1);

    // copy (n1, A2)
    cnt_n1A2 = copyGrids3D(treenodes, n1, A2, treeedges, edge_n1A2, gridsX_n1A2, gridsY_n1A2, gridsL_n1A2);

    // copy all the grids on (C1, C2) to gridsX_C1C2[] and gridsY_C1C2[]
    cnt_C1C2 = copyGrids3D(treenodes, C1, C2, treeedges, edge_C1C2, gridsX_C1C2, gridsY_C1C2,gridsL_C1C2);


    // combine grids on original (A1, n1) and (n1, A2) to new (A1, A2)
    // allocate memory for gridsX[] and gridsY[] of edge_A1A2
    if(treeedges[edge_A1A2].route.type==MAZEROUTE)
    {
        free(treeedges[edge_A1A2].route.gridsX);
        free(treeedges[edge_A1A2].route.gridsY);
		free(treeedges[edge_A1A2].route.gridsL);
    }
    len_A1A2 = cnt_n1A1+cnt_n1A2-1;

	if (len_A1A2 == 1) {
		treeedges[edge_A1A2].route.routelen = len_A1A2-1;
		treeedges[edge_A1A2].len = ADIFF(A1x,A2x)+ADIFF(A1y,A2y);
	} else {

		extraLen = 0;
		if (cnt_n1A1 > 1 && cnt_n1A2 > 1) {
			extraLen = ADIFF(gridsL_n1A1[cnt_n1A1-1],gridsL_n1A2[0]) ;
			len_A1A2 += extraLen;
		}
		treeedges[edge_A1A2].route.gridsX = (short*)calloc(len_A1A2, sizeof(short));
		treeedges[edge_A1A2].route.gridsY = (short*)calloc(len_A1A2, sizeof(short));
		treeedges[edge_A1A2].route.gridsL = (short*)calloc(len_A1A2, sizeof(short));
		treeedges[edge_A1A2].route.routelen = len_A1A2-1;
		treeedges[edge_A1A2].len = ADIFF(A1x,A2x)+ADIFF(A1y,A2y);

		cnt = 0;
		startIND = 0;

		if (cnt_n1A1 > 1 ) {
			startIND = 1;
			for(i=0; i<cnt_n1A1; i++)
			{
				treeedges[edge_A1A2].route.gridsX[cnt] = gridsX_n1A1[i];
				treeedges[edge_A1A2].route.gridsY[cnt] = gridsY_n1A1[i];
				treeedges[edge_A1A2].route.gridsL[cnt] = gridsL_n1A1[i];
				cnt++;
			}
		}

		if (extraLen > 0) {
			if (gridsL_n1A1[cnt_n1A1-1] < gridsL_n1A2[0]) {
				for (i = gridsL_n1A1[cnt_n1A1-1]+1; i <= gridsL_n1A2[0]; i++){
					treeedges[edge_A1A2].route.gridsX[cnt] = gridsX_n1A2[0];
					treeedges[edge_A1A2].route.gridsY[cnt] = gridsY_n1A2[0];
					treeedges[edge_A1A2].route.gridsL[cnt] = i;
					cnt++;
				}
			} else {
				for (i = gridsL_n1A1[cnt_n1A1-1]-1; i >= gridsL_n1A2[1]; i--){
					treeedges[edge_A1A2].route.gridsX[cnt] = gridsX_n1A2[0];
					treeedges[edge_A1A2].route.gridsY[cnt] = gridsY_n1A2[0];
					treeedges[edge_A1A2].route.gridsL[cnt] = i;
					cnt++;
				}
			}
		}


		for(i=startIND; i<cnt_n1A2; i++) // do not repeat point n1
		{
			treeedges[edge_A1A2].route.gridsX[cnt] = gridsX_n1A2[i];
			treeedges[edge_A1A2].route.gridsY[cnt] = gridsY_n1A2[i];
			treeedges[edge_A1A2].route.gridsL[cnt] = gridsL_n1A2[i];
			cnt++;
		}
	}

	if (cnt_C1C2 == 1) {
		printf("shift to 0 length edge, type2\n");
	}
  
    // find the index of E1 in (C1, C2)
    for(i=0; i<cnt_C1C2; i++)
    {
        if(gridsX_C1C2[i]==E1x && gridsY_C1C2[i]==E1y )
        {
            E1_pos1 = i;
            break;
        }
    }

	for(i=cnt_C1C2-1; i>=0; i--)
    {
        if(gridsX_C1C2[i]==E1x && gridsY_C1C2[i]==E1y )
        {
            E1_pos2 = i;
            break;
        }
    }

    // allocate memory for gridsX[] and gridsY[] of edge_n1C1 and edge_n1C2
    if(treeedges[edge_n1C1].route.type==MAZEROUTE &&treeedges[edge_n1C1].route.routelen > 0 )
    {
        free(treeedges[edge_n1C1].route.gridsX);
        free(treeedges[edge_n1C1].route.gridsY);
		free(treeedges[edge_n1C1].route.gridsL);
    }
    len_n1C1 = E1_pos1+1;

    treeedges[edge_n1C1].route.gridsX = (short*)calloc(len_n1C1, sizeof(short));
    treeedges[edge_n1C1].route.gridsY = (short*)calloc(len_n1C1, sizeof(short));
	treeedges[edge_n1C1].route.gridsL = (short*)calloc(len_n1C1, sizeof(short));
    treeedges[edge_n1C1].route.routelen = len_n1C1-1;
	treeedges[edge_n1C1].len = ADIFF(C1x,E1x)+ADIFF(C1y,E1y);

    
    if(treeedges[edge_n1C2].route.type==MAZEROUTE &&treeedges[edge_n1C2].route.routelen > 0 )
    {
        free(treeedges[edge_n1C2].route.gridsX);
        free(treeedges[edge_n1C2].route.gridsY);
		free(treeedges[edge_n1C2].route.gridsL);
    }
    len_n1C2 = cnt_C1C2-E1_pos2;

    treeedges[edge_n1C2].route.gridsX = (short*)calloc(len_n1C2, sizeof(short));
    treeedges[edge_n1C2].route.gridsY = (short*)calloc(len_n1C2, sizeof(short));
	treeedges[edge_n1C2].route.gridsL = (short*)calloc(len_n1C2, sizeof(short));
    treeedges[edge_n1C2].route.routelen = len_n1C2-1;
	treeedges[edge_n1C2].len = ADIFF(C2x,E1x)+ADIFF(C2y,E1y);


    // split original (C1, C2) to (C1, n1) and (n1, C2)
    cnt = 0;
    for(i=0; i<=E1_pos1; i++)
    {
        treeedges[edge_n1C1].route.gridsX[i] = gridsX_C1C2[i];
        treeedges[edge_n1C1].route.gridsY[i] = gridsY_C1C2[i];
		treeedges[edge_n1C1].route.gridsL[i] = gridsL_C1C2[i];
        cnt++;
    }
    ///if(cnt!=len_n1C1) {printf("len_n1C1 wrong!\n");exit(1);}
    
    cnt = 0;
    for(i=E1_pos2; i<cnt_C1C2; i++)
    {
        treeedges[edge_n1C2].route.gridsX[cnt] = gridsX_C1C2[i];
        treeedges[edge_n1C2].route.gridsY[cnt] = gridsY_C1C2[i];
		treeedges[edge_n1C2].route.gridsL[cnt] = gridsL_C1C2[i];
        cnt++;
    }
	
}






void mazeRouteMSMDOrder3D(int expand, int ripupTHlb, int ripupTHub){

	short *gridsLtmp, gridsX[MAXLEN], gridsY[MAXLEN],gridsL[MAXLEN], tmp_gridsX[MAXLEN], tmp_gridsY[MAXLEN], tmp_gridsL[MAXLEN];
    int l,  netID, enlarge, startIND, endIND;
    Bool *pop_heap23D;

    int i, j, k, deg, n1, n2, n1x, n1y, n2x, n2y, ymin, ymax, xmin, xmax, curX, curY,curL, crossX, crossY,crossL, tmpX, tmpY,tmpL, tmpi, min_x, min_y,  *dtmp;
    int segWidth, segHeight, regionX1, regionX2, regionY1, regionY2, regionWidth, regionHeight, routeLen;
    int heapLen1, heapLen2, ind, ind1, ind2, tmpind, grid;
    float costL1, costL2, tmp ;
    TreeEdge *treeedges, *treeedge, *curedge;
    TreeNode *treenodes;
    
    int endpt1, endpt2, A1, A2, B1, B2,  C1, C2, cnt, cnt_n1n2, remd;
    int edge_n1n2, edge_n1A1, edge_n1A2, edge_n1C1, edge_n1C2, edge_A1A2, edge_C1C2;
    int edge_n2B1, edge_n2B2, edge_n2D1, edge_n2D2, edge_B1B2, edge_D1D2, D1, D2;
    int E1x, E1y, E1l, E2x, E2y, E2l,range, corE1, corE2, edgeID;

	int temp_dist, tmp_of, temp_cost;
	Bool Horizontal, test, n1Shift, n2Shift, redundant;
	int lastL,origL, headRoom, tailRoom, newcnt_n1n2, numpoints, d, n1a, n2a, connectionCNT;
	int  origEng, orderIndex;
	
	directions3D = (dirctionT***)calloc(numLayers, sizeof(dirctionT**));
	corrEdge3D = (int***)calloc(numLayers, sizeof(int**));
	pr3D = (parent3D***)calloc(numLayers, sizeof(parent3D**));

	for (i = 0 ; i < numLayers; i++) {
		directions3D[i] = (dirctionT**)calloc(yGrid, sizeof(dirctionT*));
		corrEdge3D[i] = (int**)calloc(yGrid, sizeof(int*));
		pr3D[i] = (parent3D**)calloc(yGrid, sizeof(parent3D*));

		for (j = 0; j < yGrid; j++) {
			directions3D[i][j] = (dirctionT*)calloc(xGrid, sizeof(dirctionT));
			corrEdge3D[i][j] = (int*)calloc(xGrid, sizeof(int));
			pr3D[i][j] = (parent3D*)calloc(xGrid, sizeof(parent3D));
		}
	}

	pop_heap23D = (Bool*)calloc(numLayers*YRANGE*XRANGE, sizeof(Bool));

    // allocate memory for priority queue
    heap13D = (int**)calloc((yGrid*xGrid*numLayers), sizeof(int*));
    heap23D = (short**)calloc((yGrid*xGrid*numLayers), sizeof(short*));

    //cout << heap13D << endl;

	for(i=0; i<yGrid; i++) {
		for(j=0; j<xGrid; j++) {
			inRegion[i][j] = FALSE;
		}
	}
	
	range = YRANGE*XRANGE*numLayers;
	for(i=0; i<range; i++)
	{
		pop_heap23D[i] = FALSE;

	}

	startIND = numValidNets/3;
	endIND = numValidNets * 0.9;

	for(orderIndex = 0; orderIndex<endIND; orderIndex++)    {



		netID = treeOrderPV[orderIndex].treeIndex;

		//printf("netID %d\n",netID);
		//fflush(stdout);
		//if (netID == 53757)
		//{
		//	continue;
		//}

		enlarge = expand;
		deg = sttrees[netID].deg;
		treeedges = sttrees[netID].edges;
		treenodes = sttrees[netID].nodes;
		origEng = enlarge;

		for(edgeID=0; edgeID<2*deg-3; edgeID++)    {
			treeedge = &(treeedges[edgeID]);

			if (treeedge->len < ripupTHub && treeedge->len > ripupTHlb) {

				test = TRUE;
			
				n1 = treeedge->n1;
				n2 = treeedge->n2;
				n1x = treenodes[n1].x;
				n1y = treenodes[n1].y;
				n2x = treenodes[n2].x;
				n2y = treenodes[n2].y;
				routeLen = treeedges[edgeID].route.routelen;

				if(n1y<=n2y)
				{
					ymin = n1y;
					ymax = n2y;
				} else {
					ymin = n2y;
					ymax = n1y;
				}

				if(n1x<=n2x)
				{
					xmin = n1x;
					xmax = n2x;
				} else {
					xmin = n2x;
					xmax = n1x;
				}

			// ripup the routing for the edge
				if(newRipup3DType3(netID, edgeID))
				{
					enlarge = min(origEng, treeedge->route.routelen );
					
					segWidth = xmax - xmin;
					segHeight = ymax - ymin;
					regionX1 = max(0, xmin - enlarge);
					regionX2 = min(xGrid-1, xmax + enlarge);
					regionY1 = max(0, ymin - enlarge);
					regionY2 = min(yGrid-1, ymax + enlarge);
					regionWidth = regionX2 - regionX1 + 1;
					regionHeight = regionY2 - regionY1 + 1;

					n1Shift = FALSE;
					n2Shift = FALSE;
					n1a = treeedge->n1a;
					n2a = treeedge->n2a;

					// initialize pop_heap13D[] and pop_heap23D[] as FALSE (for detecting the shortest path is found or not)
					
	
					for (k = 0; k < numLayers; k++) {
						for(i=regionY1; i<=regionY2; i++)
						{
							for(j=regionX1; j<=regionX2; j++)
							{
								d13D[k][i][j] = BIG_INT;
								d23D[k][i][j] = 256;
							}
						}
					}

					// setup heap13D, heap23D and initialize d13D[][] and d23D[][] for all the grids on the two subtrees 
					setupHeap3D(netID, edgeID,  &heapLen1, &heapLen2, regionX1, regionX2, regionY1, regionY2);

					// while loop to find shortest path
					ind1 = (heap13D[0]-(int*)d13D);
					ind2 = (heap23D[0]-(short*)d23D);

					for(i=0; i<heapLen2; i++)
						pop_heap23D[(heap23D[i]-(short*)d23D)] = TRUE;


					while( pop_heap23D[ind1]==FALSE) // stop until the grid position been popped out from both heap13D and heap23D
					{
						// relax all the adjacent grids within the enlarged region for source subtree
						curL = ind1/(gridHV);
						remd = ind1%(gridHV);
						curX = remd%XRANGE;
						curY = remd/XRANGE;

						extractMin3D(heap13D, heapLen1);
						//pop_heap13D[ind1] = TRUE;
						heapLen1--;
				
						if (hCapacity3D[curL]) {  
							Horizontal = TRUE;
						} else {
							Horizontal = FALSE;
						}

						if (Horizontal) {

							// left
							if(curX>regionX1 && directions3D[curL][curY][curX] != EAST )
							{
								grid = gridHs[curL] + curY*(xGrid-1)+curX-1;
								tmp = d13D[curL][curY][curX] + 1;
								if ( h_edges3D[grid].usage<h_edges3D[grid].cap) {
									
									tmpX = curX - 1; // the left neighbor

									if(d13D[curL][curY][tmpX]>=BIG_INT) // left neighbor not been put into heap13D
									{
										d13D[curL][curY][tmpX] = tmp;
										pr3D[curL][curY][tmpX].l = curL;
										pr3D[curL][curY][tmpX].x = curX;
										pr3D[curL][curY][tmpX].y = curY;
										directions3D[curL][curY][tmpX] = WEST;
										heap13D[heapLen1] = &(d13D[curL][curY][tmpX]);
										heapLen1++;
										updateHeap3D(heap13D, heapLen1, heapLen1-1);
									} else if(d13D[curL][curY][tmpX]>tmp) // left neighbor been put into heap13D but needs update
									{
										d13D[curL][curY][tmpX] = tmp;
										pr3D[curL][curY][tmpX].l = curL;
										pr3D[curL][curY][tmpX].x = curX;
										pr3D[curL][curY][tmpX].y = curY;
										directions3D[curL][curY][tmpX] = WEST;
										dtmp = &(d13D[curL][curY][tmpX]);
										ind = 0;
										while(heap13D[ind]!=dtmp)
											ind++;
										updateHeap3D(heap13D, heapLen1, ind);
									}
								}
							}
							//right
							if(Horizontal && curX<regionX2 && directions3D[curL][curY][curX] != WEST)
							{
								grid = gridHs[curL] + curY*(xGrid-1)+curX;

								tmp = d13D[curL][curY][curX] + 1;
								tmpX = curX + 1; // the right neighbor

								if ( h_edges3D[grid].usage<h_edges3D[grid].cap) {
									if(d13D[curL][curY][tmpX]>=BIG_INT) // right neighbor not been put into heap13D
									{
										d13D[curL][curY][tmpX] = tmp;
										pr3D[curL][curY][tmpX].l = curL;
										pr3D[curL][curY][tmpX].x = curX;
										pr3D[curL][curY][tmpX].y = curY;
										directions3D[curL][curY][tmpX] = EAST;
										heap13D[heapLen1] = &(d13D[curL][curY][tmpX]);
										heapLen1++;
										updateHeap3D(heap13D, heapLen1, heapLen1-1);
									}
									else if(d13D[curL][curY][tmpX]>tmp) // right neighbor been put into heap13D but needs update
									{
										d13D[curL][curY][tmpX] = tmp;
										pr3D[curL][curY][tmpX].l = curL;
										pr3D[curL][curY][tmpX].x = curX;
										pr3D[curL][curY][tmpX].y = curY;
										directions3D[curL][curY][tmpX] = EAST;
										dtmp = &(d13D[curL][curY][tmpX]);
										ind = 0;
										while(heap13D[ind]!=dtmp)
											ind++;
										updateHeap3D(heap13D, heapLen1, ind);
									}
								}
							}
						} else {
							//bottom
							if(!Horizontal && curY>regionY1 && directions3D[curL][curY][curX] != SOUTH)
							{
								grid = gridVs[curL]+(curY-1)*xGrid+curX;
								tmp = d13D[curL][curY][curX] + 1;
								tmpY = curY - 1; // the bottom neighbor
								if (v_edges3D[grid].usage < v_edges3D[grid].cap) {
									
									if(d13D[curL][tmpY][curX]>=BIG_INT) // bottom neighbor not been put into heap13D
									{
										d13D[curL][tmpY][curX] = tmp;
										pr3D[curL][tmpY][curX].l = curL;
										pr3D[curL][tmpY][curX].x = curX;
										pr3D[curL][tmpY][curX].y = curY;
										directions3D[curL][tmpY][curX] = NORTH;
										heap13D[heapLen1] = &(d13D[curL][tmpY][curX]);
										heapLen1++;
										updateHeap3D(heap13D, heapLen1, heapLen1-1);
									}
									else if(d13D[curL][tmpY][curX]>tmp) // bottom neighbor been put into heap13D but needs update
									{
										d13D[curL][tmpY][curX] = tmp;
										pr3D[curL][tmpY][curX].l = curL;
										pr3D[curL][tmpY][curX].x = curX;
										pr3D[curL][tmpY][curX].y = curY;
										directions3D[curL][tmpY][curX] = NORTH;
										dtmp = &(d13D[curL][tmpY][curX]);
										ind = 0;
										while(heap13D[ind]!=dtmp)
											ind++;
										updateHeap3D(heap13D, heapLen1, ind);
									}
								}
							}
							//top
							if(!Horizontal && curY<regionY2 && directions3D[curL][curY][curX] != NORTH)
							{
								grid = gridVs[curL] + curY*xGrid+curX;
								tmp = d13D[curL][curY][curX] + 1;
								tmpY = curY + 1; // the top neighbor
								if ( v_edges3D[grid].usage < v_edges3D[grid].cap) {
									
									if(d13D[curL][tmpY][curX]>=BIG_INT) // top neighbor not been put into heap13D
									{
										d13D[curL][tmpY][curX] = tmp;
										pr3D[curL][tmpY][curX].l = curL;
										pr3D[curL][tmpY][curX].x = curX;
										pr3D[curL][tmpY][curX].y = curY;
										directions3D[curL][tmpY][curX] = SOUTH;
										heap13D[heapLen1] = &(d13D[curL][tmpY][curX]);
										heapLen1++;
										updateHeap3D(heap13D, heapLen1, heapLen1-1);
									}
									else if(d13D[curL][tmpY][curX]>tmp) // top neighbor been put into heap13D but needs update
									{
										d13D[curL][tmpY][curX] = tmp;
										pr3D[curL][tmpY][curX].l = curL;
										pr3D[curL][tmpY][curX].x = curX;
										pr3D[curL][tmpY][curX].y = curY;
										directions3D[curL][tmpY][curX] = SOUTH;
										dtmp = &(d13D[curL][tmpY][curX]);
										ind = 0;
										while(heap13D[ind]!=dtmp)
											ind++;
										updateHeap3D(heap13D, heapLen1, ind);
									}
								}
							}
						}

						//down
						if(curL>0 && directions3D[curL][curY][curX] != UP)
						{
						
							tmp = d13D[curL][curY][curX] + viacost;
							tmpL = curL - 1; // the bottom neighbor

							//printf("down, new value %f, old value %f\n",tmp,d13D[tmpL][curY][curX]);
							if(d13D[tmpL][curY][curX]>=BIG_INT) // bottom neighbor not been put into heap13D
							{
								d13D[tmpL][curY][curX] = tmp;
								pr3D[tmpL][curY][curX].l = curL;
								pr3D[tmpL][curY][curX].x = curX;
								pr3D[tmpL][curY][curX].y = curY;
								directions3D[tmpL][curY][curX] = DOWN;
								heap13D[heapLen1] = &(d13D[tmpL][curY][curX]);
								heapLen1++;
								updateHeap3D(heap13D, heapLen1, heapLen1-1);
							}
							else if(d13D[tmpL][curY][curX]>tmp) // bottom neighbor been put into heap13D but needs update
							{
								d13D[tmpL][curY][curX] = tmp;
								pr3D[tmpL][curY][curX].l = curL;
								pr3D[tmpL][curY][curX].x = curX;
								pr3D[tmpL][curY][curX].y = curY;
								directions3D[tmpL][curY][curX] = DOWN;
								dtmp = &(d13D[tmpL][curY][curX]);
								ind = 0;
								while(heap13D[ind]!=dtmp)
									ind++;
								updateHeap3D(heap13D, heapLen1, ind);
							}
						}

						//up
						if(curL < numLayers - 1 && directions3D[curL][curY][curX] != DOWN)
						{
						
							tmp = d13D[curL][curY][curX] + viacost;
							tmpL = curL + 1; // the bottom neighbor
							if(d13D[tmpL][curY][curX]>=BIG_INT) // bottom neighbor not been put into heap13D
							{
								d13D[tmpL][curY][curX] = tmp;
								pr3D[tmpL][curY][curX].l = curL;
								pr3D[tmpL][curY][curX].x = curX;
								pr3D[tmpL][curY][curX].y = curY;
								directions3D[tmpL][curY][curX] = UP;
								heap13D[heapLen1] = &(d13D[tmpL][curY][curX]);
								heapLen1++;
								updateHeap3D(heap13D, heapLen1, heapLen1-1);
							}
							else if(d13D[tmpL][curY][curX]>tmp) // bottom neighbor been put into heap13D but needs update
							{
								d13D[tmpL][curY][curX] = tmp;
								pr3D[tmpL][curY][curX].l = curL;
								pr3D[tmpL][curY][curX].x = curX;
								pr3D[tmpL][curY][curX].y = curY;
								directions3D[tmpL][curY][curX] = UP;
								dtmp = &(d13D[tmpL][curY][curX]);
								ind = 0;
								while(heap13D[ind]!=dtmp)
									ind++;
								updateHeap3D(heap13D, heapLen1, ind);
							}
						}

						// update ind1 and ind2 for next loop
						ind1 = (heap13D[0]-(int*)d13D);
					} // while loop

			
					for(i=0; i<heapLen2; i++)
						pop_heap23D[(heap23D[i]-(short*)d23D)] = FALSE;


					// get the new route for the edge and store it in gridsX[] and gridsY[] temporarily

					crossL = ind1/(gridHV);
					crossX = (ind1%(gridHV))%XRANGE;
					crossY = (ind1%(gridHV))/XRANGE;

					cnt = 0;
					curX = crossX;
					curY = crossY;
					curL = crossL;

					if (d13D[curL][curY][curX] == 0) {
						recoverEdge(netID, edgeID);
						break;
					}
					//printf("the initial value %f LYX [%d %d %d]\n",d13D[curL][curY][curX],curL, curY, curX);

					while(d13D[curL][curY][curX]!=0) // loop until reach subtree1
					{
						
						tmpL = pr3D[curL][curY][curX].l;
						tmpX = pr3D[curL][curY][curX].x;
						tmpY = pr3D[curL][curY][curX].y;
						curX = tmpX;
						curY = tmpY;
						curL = tmpL;
						fflush(stdout);
						tmp_gridsX[cnt] = curX;
						tmp_gridsY[cnt] = curY;
						tmp_gridsL[cnt] = curL;
						cnt++;
					}

					//printf("the end value %f\n",d13D[curL][curY][curX]);
					// reverse the grids on the path
					for(i=0; i<cnt; i++)
					{
						tmpind = cnt-1-i;
						gridsX[i] = tmp_gridsX[tmpind];
						gridsY[i] = tmp_gridsY[tmpind];
						gridsL[i] = tmp_gridsL[tmpind];
					}

					// add the connection point (crossX, crossY)
					gridsX[cnt] = crossX;
					gridsY[cnt] = crossY;
					gridsL[cnt] = crossL;
					cnt++;

					curX = crossX;
					curY = crossY;
					curL = crossL;

					cnt_n1n2 = cnt;

					E1x = gridsX[0];
					E1y = gridsY[0];
					E2x = gridsX[cnt_n1n2-1];
					E2y = gridsY[cnt_n1n2-1];

					


					headRoom = 0;
					origL =  gridsL[0];

					while (gridsX[headRoom] == E1x && gridsY[headRoom] == E1y) {
						lastL = gridsL[headRoom];
						headRoom++;
					}
					if (headRoom > 0) {
						headRoom --;
					}

					lastL =  gridsL[headRoom];

			
					// change the tree structure according to the new routing for the tree edge
					// find E1 and E2, and the endpoints of the edges they are on
	
					edge_n1n2 = edgeID;
					// (1) consider subtree1
					if(n1>=deg && (E1x!=n1x || E1y!=n1y))
					// n1 is not a pin and E1!=n1, then make change to subtree1, otherwise, no change to subtree1
					{
						n1Shift = TRUE;
						corE1 = corrEdge3D[origL][E1y][E1x];

						endpt1 = treeedges[corE1].n1;
						endpt2 = treeedges[corE1].n2;

						// find A1, A2 and edge_n1A1, edge_n1A2
						if(treenodes[n1].nbr[0]==n2)
						{
							A1 = treenodes[n1].nbr[1];
							A2 = treenodes[n1].nbr[2];
							edge_n1A1 = treenodes[n1].edge[1];
							edge_n1A2 = treenodes[n1].edge[2];
						}
						else if(treenodes[n1].nbr[1]==n2)
						{
							A1 = treenodes[n1].nbr[0];
							A2 = treenodes[n1].nbr[2];
							edge_n1A1 = treenodes[n1].edge[0];
							edge_n1A2 = treenodes[n1].edge[2];
						}
						else
						{
							A1 = treenodes[n1].nbr[0];
							A2 = treenodes[n1].nbr[1];
							edge_n1A1 = treenodes[n1].edge[0];
							edge_n1A2 = treenodes[n1].edge[1];
						}

						if(endpt1==n1 || endpt2==n1) // E1 is on (n1, A1) or (n1, A2)
						{
							// if E1 is on (n1, A2), switch A1 and A2 so that E1 is always on (n1, A1)
							if(endpt1==A2 || endpt2==A2)
							{
								tmpi = A1;
								A1 = A2;
								A2 = tmpi;
								tmpi = edge_n1A1;
								edge_n1A1 = edge_n1A2;
								edge_n1A2 = tmpi;
							}

							// update route for edge (n1, A1), (n1, A2)
							updateRouteType13D(netID, treenodes, n1, A1, A2, E1x, E1y, treeedges, edge_n1A1, edge_n1A2);
							//newUpdateNodeLayers(treenodes, edge_n1n2,n1, lastL); 

							// update position for n1
							
							//treenodes[n1].l = E1l;
							treenodes[n1].assigned = TRUE;
						} // if E1 is on (n1, A1) or (n1, A2)
						else // E1 is not on (n1, A1) or (n1, A2), but on (C1, C2)
						{
							C1 = endpt1;
							C2 = endpt2;
							edge_C1C2 = corrEdge3D[origL][E1y][E1x];
							
							// update route for edge (n1, C1), (n1, C2) and (A1, A2)
							updateRouteType23D(netID,treenodes, n1, A1, A2, C1, C2, E1x, E1y,treeedges, edge_n1A1, edge_n1A2, edge_C1C2);
							// update position for n1
							treenodes[n1].x = E1x;
							treenodes[n1].y = E1y;
							treenodes[n1].assigned = TRUE;
							// update 3 edges (n1, A1)->(C1, n1), (n1, A2)->(n1, C2), (C1, C2)->(A1, A2)
							edge_n1C1 = edge_n1A1;
							treeedges[edge_n1C1].n1 = C1;
							treeedges[edge_n1C1].n2 = n1;
							edge_n1C2 = edge_n1A2;
							treeedges[edge_n1C2].n1 = n1;
							treeedges[edge_n1C2].n2 = C2;
							edge_A1A2 = edge_C1C2;
							treeedges[edge_A1A2].n1 = A1;
							treeedges[edge_A1A2].n2 = A2;
							// update nbr and edge for 5 nodes n1, A1, A2, C1, C2
							// n1's nbr (n2, A1, A2)->(n2, C1, C2)
							treenodes[n1].nbr[0] = n2;
							treenodes[n1].edge[0] = edge_n1n2;
							treenodes[n1].nbr[1] = C1;
							treenodes[n1].edge[1] = edge_n1C1;
							treenodes[n1].nbr[2] = C2;
							treenodes[n1].edge[2] = edge_n1C2;
							// A1's nbr n1->A2
							for(i=0; i<3; i++)
							{
								if(treenodes[A1].nbr[i]==n1)
								{
									treenodes[A1].nbr[i] = A2;
									treenodes[A1].edge[i] = edge_A1A2;
									break;
								}
							}
							// A2's nbr n1->A1
							for(i=0; i<3; i++)
							{
								if(treenodes[A2].nbr[i]==n1)
								{
									treenodes[A2].nbr[i] = A1;
									treenodes[A2].edge[i] = edge_A1A2;
									break;
								}
							}
							// C1's nbr C2->n1
							for(i=0; i<3; i++)
							{
								if(treenodes[C1].nbr[i]==C2)
								{
									treenodes[C1].nbr[i] = n1;
									treenodes[C1].edge[i] = edge_n1C1;
									break;
								}
							}
							// C2's nbr C1->n1
							for(i=0; i<3; i++)
							{
								if(treenodes[C2].nbr[i]==C1)
								{
									treenodes[C2].nbr[i] = n1;
									treenodes[C2].edge[i] = edge_n1C2;
									break;
								}
							}
						} // else E1 is not on (n1, A1) or (n1, A2), but on (C1, C2)
					} // n1 is not a pin and E1!=n1
					else 
					{
						newUpdateNodeLayers(treenodes, edge_n1n2,n1a, lastL); 
					}

					
					origL = gridsL[cnt_n1n2-1];
					tailRoom = cnt_n1n2-1;

					while (gridsX[tailRoom] == E2x && gridsY[tailRoom] == E2y) {
						tailRoom--;
					}
					if (tailRoom < cnt_n1n2-1) {
						tailRoom ++;
					}

					lastL = gridsL[tailRoom];


					//updateNodeLayers(treenodes, edgeID, n2a, ntpL,nbtL,lastL);

					// (2) consider subtree2
					if(n2>=deg && (E2x!=n2x || E2y!=n2y ))
					// n2 is not a pin and E2!=n2, then make change to subtree2, otherwise, no change to subtree2
					{
						// find the endpoints of the edge E1 is on

						n2Shift = TRUE;
						corE2 = corrEdge3D[origL][E2y][E2x];
						endpt1 = treeedges[corE2].n1;
						endpt2 = treeedges[corE2].n2;

						// find B1, B2
						if(treenodes[n2].nbr[0]==n1)
						{
							B1 = treenodes[n2].nbr[1];
							B2 = treenodes[n2].nbr[2];
							edge_n2B1 = treenodes[n2].edge[1];
							edge_n2B2 = treenodes[n2].edge[2];
						}
						else if(treenodes[n2].nbr[1]==n1)
						{
							B1 = treenodes[n2].nbr[0];
							B2 = treenodes[n2].nbr[2];
							edge_n2B1 = treenodes[n2].edge[0];
							edge_n2B2 = treenodes[n2].edge[2];
						}
						else
						{
							B1 = treenodes[n2].nbr[0];
							B2 = treenodes[n2].nbr[1];
							edge_n2B1 = treenodes[n2].edge[0];
							edge_n2B2 = treenodes[n2].edge[1];
						}

						if(endpt1==n2 || endpt2==n2) // E2 is on (n2, B1) or (n2, B2)
						{
							// if E2 is on (n2, B2), switch B1 and B2 so that E2 is always on (n2, B1)
							if(endpt1==B2 || endpt2==B2)
							{
								tmpi = B1;
								B1 = B2;
								B2 = tmpi;
								tmpi = edge_n2B1;
								edge_n2B1 = edge_n2B2;
								edge_n2B2 = tmpi;
							}
							//printf(" type1\n");

							// update route for edge (n2, B1), (n2, B2)
							updateRouteType13D(netID,treenodes, n2, B1, B2, E2x, E2y, treeedges, edge_n2B1, edge_n2B2);
							//newUpdateNodeLayers(treenodes, edge_n1n2,n2, lastL); 


							// update position for n2
							treenodes[n2].assigned = TRUE;
						} // if E2 is on (n2, B1) or (n2, B2)
						else // E2 is not on (n2, B1) or (n2, B2), but on (d13D, d23D)
						{
							D1 = endpt1;
							D2 = endpt2;
							edge_D1D2 = corrEdge3D[origL][E2y][E2x];
							//printf(" type2\n");

							// update route for edge (n2, d13D), (n2, d23D) and (B1, B2)
							updateRouteType23D(netID,treenodes, n2, B1, B2, D1, D2, E2x, E2y,  treeedges, edge_n2B1, edge_n2B2, edge_D1D2);
							// update position for n2
							treenodes[n2].x = E2x;
							treenodes[n2].y = E2y;
							treenodes[n2].assigned = TRUE;
							// update 3 edges (n2, B1)->(d13D, n2), (n2, B2)->(n2, d23D), (d13D, d23D)->(B1, B2)
							edge_n2D1 = edge_n2B1;
							treeedges[edge_n2D1].n1 = D1;
							treeedges[edge_n2D1].n2 = n2;
							edge_n2D2 = edge_n2B2;
							treeedges[edge_n2D2].n1 = n2;
							treeedges[edge_n2D2].n2 = D2;
							edge_B1B2 = edge_D1D2;
							treeedges[edge_B1B2].n1 = B1;
							treeedges[edge_B1B2].n2 = B2;
							// update nbr and edge for 5 nodes n2, B1, B2, d13D, d23D
							// n1's nbr (n1, B1, B2)->(n1, d13D, d23D)
							treenodes[n2].nbr[0] = n1;
							treenodes[n2].edge[0] = edge_n1n2;
							treenodes[n2].nbr[1] = D1;
							treenodes[n2].edge[1] = edge_n2D1;
							treenodes[n2].nbr[2] = D2;
							treenodes[n2].edge[2] = edge_n2D2;
							// B1's nbr n2->B2
							for(i=0; i<3; i++)
							{
								if(treenodes[B1].nbr[i]==n2)
								{
									treenodes[B1].nbr[i] = B2;
									treenodes[B1].edge[i] = edge_B1B2;
									break;
								}
							}
							// B2's nbr n2->B1
							for(i=0; i<3; i++)
							{
								if(treenodes[B2].nbr[i]==n2)
								{
									treenodes[B2].nbr[i] = B1;
									treenodes[B2].edge[i] = edge_B1B2;
									break;
								}
							}
							// D1's nbr D2->n2
							for(i=0; i<3; i++)
							{
								if(treenodes[D1].nbr[i]==D2)
								{
									treenodes[D1].nbr[i] = n2;
									treenodes[D1].edge[i] = edge_n2D1;
									break;
								}
							}
							// D2's nbr D1->n2
							for(i=0; i<3; i++)
							{
								if(treenodes[D2].nbr[i]==D1)
								{
									treenodes[D2].nbr[i] = n2;
									treenodes[D2].edge[i] = edge_n2D2;
									break;
								}
							}
						} // else E2 is not on (n2, B1) or (n2, B2), but on (d13D, d23D)
					} else // n2 is not a pin and E2!=n2
					{
						newUpdateNodeLayers(treenodes, edge_n1n2,n2a, lastL); 
					}


					newcnt_n1n2 = tailRoom - headRoom +1;

					// update route for edge (n1, n2) and edge usage
					if(treeedges[edge_n1n2].route.type==MAZEROUTE)
					{
						free(treeedges[edge_n1n2].route.gridsX);
						free(treeedges[edge_n1n2].route.gridsY);
						free(treeedges[edge_n1n2].route.gridsL);
					}

					treeedges[edge_n1n2].route.gridsX = (short*)calloc(newcnt_n1n2, sizeof(short));
					treeedges[edge_n1n2].route.gridsY = (short*)calloc(newcnt_n1n2, sizeof(short));
					treeedges[edge_n1n2].route.gridsL = (short*)calloc(newcnt_n1n2, sizeof(short));
					treeedges[edge_n1n2].route.type = MAZEROUTE;
					treeedges[edge_n1n2].route.routelen = newcnt_n1n2-1;
					treeedges[edge_n1n2].len= ADIFF(E1x,E2x)+ADIFF(E1y,E2y);

					j = headRoom;
					for(i=0; i<newcnt_n1n2; i++)
					{
						treeedges[edge_n1n2].route.gridsX[i] = gridsX[j];
						treeedges[edge_n1n2].route.gridsY[i] = gridsY[j];
						treeedges[edge_n1n2].route.gridsL[i] = gridsL[j];
						j++;
					}

					// update edge usage
					for(i=headRoom; i<tailRoom; i++)
					{
						if (gridsL[i]==gridsL[i+1]) {
							if(gridsX[i]==gridsX[i+1]) // a vertical edge
							{
								min_y = min(gridsY[i], gridsY[i+1]);
								v_edges3D[gridsL[i]*gridV+min_y*xGrid+gridsX[i]].usage += 1;
							}
							else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
							{
								min_x = min(gridsX[i], gridsX[i+1]);
								h_edges3D[gridsL[i]*gridH+gridsY[i]*(xGrid-1)+min_x].usage += 1;
							}
						}
					}


					if (n1Shift || n2Shift) {
					// re statis the node overlap
						numpoints=0;

						for(d=0;d<2*deg-2;d++)
						{
							treenodes[d].topL = -1;
							treenodes[d].botL = numLayers;
							treenodes[d].assigned = FALSE;
							treenodes[d].stackAlias = d;
							treenodes[d].conCNT = 0;
							treenodes[d].hID = BIG_INT;
							treenodes[d].lID = BIG_INT;
							treenodes[d].status = 0;


							if(d<deg)
							{
								treenodes[d].botL = treenodes[d].topL   = 0;
								//treenodes[d].l = 0;
								treenodes[d].assigned = TRUE;
								treenodes[d].status = 1;
							
								xcor[numpoints]=treenodes[d].x;
								ycor[numpoints]=treenodes[d].y;
								dcor[numpoints]=d;
								numpoints++;
							} else {
								redundant=FALSE;
								for(k=0;k<numpoints;k++)
								{
									if((treenodes[d].x==xcor[k])&&(treenodes[d].y==ycor[k]))
									{
										treenodes[d].stackAlias = dcor[k];
										
										redundant=TRUE;
										break;
									}
								}
								if(!redundant)
								{
									xcor[numpoints]=treenodes[d].x;
									ycor[numpoints]=treenodes[d].y;
									dcor[numpoints]=d;
									numpoints++;
								}
							}
						} // numerating for nodes
						for (k = 0; k < 2*deg-3; k++) {

							treeedge = &(treeedges[k]);

							if (treeedge->len > 0) {

								routeLen = treeedge->route.routelen;

								n1 = treeedge->n1;
								n2 = treeedge->n2;
								gridsLtmp = treeedge->route.gridsL;

								n1a = treenodes[n1].stackAlias;

							
								n2a = treenodes[n2].stackAlias;

								treeedge->n1a = n1a;
								treeedge->n2a = n2a;

								connectionCNT = treenodes[n1a].conCNT;
								treenodes[n1a].heights[connectionCNT] = gridsLtmp[0];
								treenodes[n1a].eID[connectionCNT] = k;
								treenodes[n1a].conCNT++;
								

								if (gridsLtmp[0]>treenodes[n1a].topL) {
									treenodes[n1a].hID = k;
									treenodes[n1a].topL = gridsLtmp[0];
								}
								if (gridsLtmp[0]<treenodes[n1a].botL) {
									treenodes[n1a].lID = k;
									treenodes[n1a].botL = gridsLtmp[0];
								}

								treenodes[n1a].assigned = TRUE;

								connectionCNT = treenodes[n2a].conCNT;
								treenodes[n2a].heights[connectionCNT] = gridsLtmp[routeLen];
								treenodes[n2a].eID[connectionCNT] = k;
								treenodes[n2a].conCNT++;
								if (gridsLtmp[routeLen]>treenodes[n2a].topL) {
									treenodes[n2a].hID = k;
									treenodes[n2a].topL = gridsLtmp[routeLen];
								} 
								if (gridsLtmp[routeLen]<treenodes[n2a].botL) {
									treenodes[n2a].lID = k;
									treenodes[n2a].botL = gridsLtmp[routeLen];
								}

								treenodes[n2a].assigned = TRUE;
							
							}//edge len > 0

						} // eunmerating edges 

					//	printf("edge %d shifted post processing finished\n",edgeID);
					} // if shift1 and shift2
										
				}
			}
		}
	}


	for (i = 0 ; i < numLayers; i++) {
		for (j = 0; j < yGrid; j++) {
			free (directions3D[i][j]);
			free (corrEdge3D[i][j]);
			free (pr3D[i][j]);
		}
	}

	for (i = 0 ; i < numLayers; i++) {

		free (directions3D[i]);
		free (corrEdge3D[i]);
		free (pr3D[i]);
	} 

	free (directions3D);
	free(corrEdge3D);
	free (pr3D);


    
    free(pop_heap23D);
    free(heap13D);
    free(heap23D);

}




void getLayerRange(TreeNode *treenodes ,int edgeID, int n1, int deg)
{
	int i,k;
	int ntpL, nbtL, nhID, nlID;

	ntpL = -1;
	nbtL = BIG_INT;

	if ( treenodes[n1].conCNT > 1) {
		for(i = 0; i < treenodes[n1].conCNT; i++) {
			if (treenodes[n1].eID[i] != edgeID) {
				if (ntpL < treenodes[n1].heights[i])  {
					ntpL = treenodes[n1].heights[i];
					nhID = treenodes[n1].eID[i];
				} 
				if (nbtL > treenodes[n1].heights[i]) {
					nbtL = treenodes[n1].heights[i];
					nlID = treenodes[n1].eID[i];
				}
			}
		}
		if (n1 < deg) {
			nbtL = 0;
		}
		treenodes[n1].topL = ntpL;
		treenodes[n1].botL = nbtL;
		treenodes[n1].hID = nhID;
		treenodes[n1].lID = nlID;
	} else {

		if (treenodes[n1].botL > 0) {
			printf("bottom layer acutally %d\n",treenodes[n1].botL);
		}
		treenodes[n1].topL = 0;
		treenodes[n1].botL = 0;
		treenodes[n1].hID = BIG_INT;
		treenodes[n1].lID = BIG_INT;
		if (n1>=deg) {
			printf("steiner nodes only have one connection\n");
			exit(0);
		}
	}
}

#endif
