#ifndef _EDGESHIFT_H_
#define _EDGESHIFT_H_

#include <stdio.h>
#include <stdlib.h>
#include "DataType.h"
#include "flute.h"
#include "DataProc.h"
#include "route.h"
#include "RipUp.h"

#define HORIZONTAL 1
#define VERTICAL 0



int edgeShift(Tree *t, int net)
{
    int i, j, k, l, m, deg, root, x, y, n, nx, ny, n1, n2, n3;
    int maxX, minX, maxY, minY, maxX1, minX1, maxY1, minY1, maxX2, minX2, maxY2, minY2, bigX, smallX, bigY, smallY, grid, grid1, grid2;
    int nbr[MAXNETDEG*2][3], nbrCnt[MAXNETDEG*2];
    int pairCnt, pairN1[MAXNETDEG], pairN2[MAXNETDEG];
    int benefit, bestBenefit, bestCost;
    int cost1, cost2, *costH, *costV, bestPair, Pos, bestPos, numShift = 0;
    
    costH = (int*)malloc(yGrid*sizeof(int));
    costV = (int*)malloc(xGrid*sizeof(int));
    
    deg = t->deg;
    // find root of the tree
    for(i=deg; i<2*deg-2; i++)
    {
        if(t->branch[i].n == i)
        {
            root = i;
            break;
        }
    }
//printf("root=%d\n", root);    

    // find all neighbors for steiner nodes
    for(i=deg; i<2*deg-2; i++)
        nbrCnt[i] = 0;
      // edges from pin to steiner
    for(i=0; i<deg; i++)
    {
        n = t->branch[i].n;
        nbr[n][nbrCnt[n]] = i;
        nbrCnt[n] ++;
    }
      // edges from steiner to steiner
    for(i=deg; i<2*deg-2; i++)
    {
        if(i!=root) // not the removed steiner nodes and root
        {
            n = t->branch[i].n;
            nbr[i][nbrCnt[i]] = n;
            nbrCnt[i] ++;
            nbr[n][nbrCnt[n]] = i;
            nbrCnt[n] ++;
        }
    }

    
//for(i=deg; i<2*deg-2; i++)
//   if(nbrCnt[i]!=3) printf("nbrCnt[%d]=%d(!=3)\n", i, nbrCnt[i]);
   
    bestBenefit = BIG_INT; // used to enter while loop
    while(bestBenefit>0)// && numShift<60)
    {
        // find all H or V edges (steiner pairs)
        pairCnt = 0;
        for(i=deg; i<2*deg-2; i++)
        {
            n = t->branch[i].n;
            if(t->branch[i].x == t->branch[n].x)
            {
                if(t->branch[i].y < t->branch[n].y)
                {
                    pairN1[pairCnt] = i;
                    pairN2[pairCnt] = n;
                    pairCnt ++;
                }
                else if(t->branch[i].y > t->branch[n].y)
                {
                    pairN1[pairCnt] = n;
                    pairN2[pairCnt] = i;
                    pairCnt ++;
                }
            }
            else if(t->branch[i].y == t->branch[n].y)
            {
                if(t->branch[i].x < t->branch[n].x)
                {
                    pairN1[pairCnt] = i;
                    pairN2[pairCnt] = n;
                    pairCnt ++;
                }
                else if(t->branch[i].x > t->branch[n].x)
                {
                    pairN1[pairCnt] = n;
                    pairN2[pairCnt] = i;
                    pairCnt ++;
                }
            }
        }
        
        bestPair = -1;
        bestBenefit = -1;
        // for each H or V edge, find the best benefit by shifting it
        for(i=0; i<pairCnt; i++)
        {
            // find the range of shifting for this pair
            n1 = pairN1[i];
            n2 = pairN2[i];
            if(t->branch[n1].y==t->branch[n2].y) // a horizontal edge
            {
                // find the shifting range for the edge (minY~maxY)
                maxY1 = minY1 = t->branch[n1].y;
                for(j=0; j<3; j++)
                {
                    y = t->branch[nbr[n1][j]].y;
                    if(y>maxY1)
                        maxY1 = y;
                    else if(y<minY1)
                        minY1 = y;
                }
                maxY2 = minY2 = t->branch[n2].y;
                for(j=0; j<3; j++)
                {
                    y = t->branch[nbr[n2][j]].y;
                    if(y>maxY2)
                        maxY2 = y;
                    else if(y<minY2)
                        minY2 = y;
                }
                minY = max(minY1, minY2);
                maxY = min(maxY1, maxY2);
//printf("(%d-%d) minY1=%d, maxY1=%d, minY2=%d, maxY2=%d, minY = %d, maxY = %d\n", n1, n2, minY1, maxY1, minY2, maxY2, minY, maxY);                

                // find the best position (least total usage) to shift 
                if(minY<maxY) // more than 1 possible positions
                {
                    for(j=minY; j<=maxY; j++)
                    {
                        costH[j] = 0;
                        grid = j*(xGrid-1);
                        for(k=t->branch[n1].x; k<t->branch[n2].x; k++)
                        {
                            costH[j] += h_edges[grid+k].est_usage;
                        }
                        // add the cost of all edges adjacent to the two steiner nodes
                        for(l=0; l<nbrCnt[n1]; l++)
                        {
                            n3 = nbr[n1][l];
                            if(n3!=n2) // exclude current edge n1-n2
                            {
                                cost1 = cost2 = 0;
                                if(t->branch[n1].x < t->branch[n3].x)
                                {
                                    smallX = t->branch[n1].x;
                                    bigX = t->branch[n3].x;
                                }
                                else
                                {
                                    smallX = t->branch[n3].x;
                                    bigX = t->branch[n1].x;
                                }
                                if(j < t->branch[n3].y)
                                {
                                    smallY = j;
                                    bigY = t->branch[n3].y;
                                }
                                else
                                {
                                    smallY = t->branch[n3].y;
                                    bigY = j;
                                }
                                grid1 = smallY*(xGrid-1);
                                grid2 = bigY*(xGrid-1);
                                for(m=smallX; m<bigX; m++)
                                {
                                    cost1 += h_edges[grid1+m].est_usage;
                                    cost2 += h_edges[grid2+m].est_usage;
                                }
                                grid1 = smallY*xGrid;
                                for(m=smallY; m<bigY; m++)
                                {
                                    cost1 += v_edges[grid1+bigX].est_usage;
                                    cost2 += v_edges[grid1+smallX].est_usage;
                                    grid1 += xGrid;
                                }
                                costH[j] += min(cost1, cost2);
                            } // if(n3!=n2)
                        } // loop l
                        for(l=0; l<nbrCnt[n2]; l++)
                        {
                            n3 = nbr[n2][l];
                            if(n3!=n1) // exclude current edge n1-n2
                            {
                                cost1 = cost2 = 0;
                                if(t->branch[n2].x < t->branch[n3].x)
                                {
                                    smallX = t->branch[n2].x;
                                    bigX = t->branch[n3].x;
                                }
                                else
                                {
                                    smallX = t->branch[n3].x;
                                    bigX = t->branch[n2].x;
                                }
                                if(j < t->branch[n3].y)
                                {
                                    smallY = j;
                                    bigY = t->branch[n3].y;
                                }
                                else
                                {
                                    smallY = t->branch[n3].y;
                                    bigY = j;
                                }
                                grid1 = smallY*(xGrid-1);
                                grid2 = bigY*(xGrid-1);
                                for(m=smallX; m<bigX; m++)
                                {
                                    cost1 += h_edges[grid1+m].est_usage;
                                    cost2 += h_edges[grid2+m].est_usage;
                                }
                                grid1 = smallY*xGrid;
                                for(m=smallY; m<bigY; m++)
                                {
                                    cost1 += v_edges[grid1+bigX].est_usage;
                                    cost2 += v_edges[grid1+smallX].est_usage;
                                    grid1 += xGrid;
                                }
                                costH[j] += min(cost1, cost2);
                            } //if(n3!=n1)
                        } // loop l
                    } // loop j
                    bestCost = BIG_INT;
                    Pos = t->branch[n1].y;
                    for(j=minY; j<=maxY; j++)
                    {
                        if(costH[j]<bestCost)
                        {
                            bestCost = costH[j];
                            Pos = j;
                        }
                    }
                    if(Pos!=t->branch[n1].y) // find a better position than current
                    {
                        benefit = costH[t->branch[n1].y]-bestCost;
                        if(benefit>bestBenefit)
                        {
                            bestBenefit = benefit;
                            bestPair = i;
                            bestPos = Pos;
                        }
                    }
                }

            }
            else // a vertical edge
            {
                // find the shifting range for the edge (minX~maxX)
                maxX1 = minX1 = t->branch[n1].x;
                for(j=0; j<3; j++)
                {
                    x = t->branch[nbr[n1][j]].x;
                    if(x>maxX1)
                        maxX1 = x;
                    else if(x<minX1)
                        minX1 = x;
                }
                maxX2 = minX2 = t->branch[n2].x;
                for(j=0; j<3; j++)
                {
                    x = t->branch[nbr[n2][j]].x;
                    if(x>maxX2)
                        maxX2 = x;
                    else if(x<minX2)
                        minX2 = x;
                }
                minX = max(minX1, minX2);
                maxX = min(maxX1, maxX2);

                // find the best position (least total usage) to shift
                if(minX<maxX) // more than 1 possible positions
                {
                    for(j=minX; j<=maxX; j++)
                    {
                        costV[j] = 0;
                        for(k=t->branch[n1].y; k<t->branch[n2].y; k++)
                        {
                            costV[j] += v_edges[k*xGrid+j].est_usage;
                        }
                        // add the cost of all edges adjacent to the two steiner nodes
                        for(l=0; l<nbrCnt[n1]; l++)
                        {
                            n3 = nbr[n1][l];
                            if(n3!=n2) // exclude current edge n1-n2
                            {
                                cost1 = cost2 = 0;
                                if(j < t->branch[n3].x)
                                {
                                    smallX = j;
                                    bigX = t->branch[n3].x;
                                }
                                else
                                {
                                    smallX = t->branch[n3].x;
                                    bigX = j;
                                }
                                if(t->branch[n1].y < t->branch[n3].y)
                                {
                                    smallY = t->branch[n1].y;
                                    bigY = t->branch[n3].y;
                                }
                                else
                                {
                                    smallY = t->branch[n3].y;
                                    bigY = t->branch[n1].y;
                                }
                                grid1 = smallY*(xGrid-1);
                                grid2 = bigY*(xGrid-1);
                                for(m=smallX; m<bigX; m++)
                                {
                                    cost1 += h_edges[grid1+m].est_usage;
                                    cost2 += h_edges[grid2+m].est_usage;
                                }
                                grid1 = smallY*xGrid;
                                for(m=smallY; m<bigY; m++)
                                {
                                    cost1 += v_edges[grid1+bigX].est_usage;
                                    cost2 += v_edges[grid1+smallX].est_usage;
                                    grid1 += xGrid;
                                }
                                costV[j] += min(cost1, cost2);
                            } // if(n3!=n2)
                        } // loop l
                        for(l=0; l<nbrCnt[n2]; l++)
                        {
                            n3 = nbr[n2][l];
                            if(n3!=n1) // exclude current edge n1-n2
                            {
                                cost1 = cost2 = 0;
                                if(j < t->branch[n3].x)
                                {
                                    smallX = j;
                                    bigX = t->branch[n3].x;
                                }
                                else
                                {
                                    smallX = t->branch[n3].x;
                                    bigX = j;
                                }
                                if(t->branch[n2].y < t->branch[n3].y)
                                {
                                    smallY = t->branch[n2].y;
                                    bigY = t->branch[n3].y;
                                }
                                else
                                {
                                    smallY = t->branch[n3].y;
                                    bigY = t->branch[n2].y;
                                }
                                grid1 = smallY*(xGrid-1);
                                grid2 = bigY*(xGrid-1);
                                for(m=smallX; m<bigX; m++)
                                {
                                    cost1 += h_edges[grid1+m].est_usage;
                                    cost2 += h_edges[grid2+m].est_usage;
                                }
                                grid1 = smallY*xGrid;
                                for(m=smallY; m<bigY; m++)
                                {
                                    cost1 += v_edges[grid1+bigX].est_usage;
                                    cost2 += v_edges[grid1+smallX].est_usage;
                                    grid1 += xGrid;
                                }
                                costV[j] += min(cost1, cost2);
                            } //if(n3!=n1)
                        } // loop l
                    } // loop j
                    bestCost = BIG_INT;
                    Pos = t->branch[n1].x;
                    for(j=minX; j<=maxX; j++)
                    {
                        if(costV[j]<bestCost)
                        {
                            bestCost = costV[j];
                            Pos = j;
                        }
                    }
                    if(Pos!=t->branch[n1].x) // find a better position than current
                    {
                        benefit = costV[t->branch[n1].x]-bestCost;
                        if(benefit>bestBenefit)
                        {
                            bestBenefit = benefit;
                            bestPair = i;
                            bestPos = Pos;
                        }
                    }
                }

            } // else (a vertical edge)

        } // loop i
        
        if(bestBenefit>0)
        {
            n1 = pairN1[bestPair];
            n2 = pairN2[bestPair];

            if(t->branch[n1].y==t->branch[n2].y) // horizontal edge
            {
                t->branch[n1].y = bestPos;
                t->branch[n2].y = bestPos;
            } // vertical edge
            else
            {
                t->branch[n1].x = bestPos;
                t->branch[n2].x = bestPos;
            }
            numShift++;
        }
    } // while(bestBenefit>0)

    free(costH);
    free(costV);

return(numShift);
}



// exchange Steiner nodes at the same position, then call edgeShift()
int edgeShiftNew(Tree *t, int net)
{
    int i, j, n;
    int deg, pairCnt, pairN1[MAXNETDEG], pairN2[MAXNETDEG], cur_pairN1, cur_pairN2;
    int N1nbrH, N1nbrV, N2nbrH, N2nbrV, bestBenefit, iter;
    int numShift;
    Bool isPair;
//printf("net[%d]\n", net); getchar();   
    numShift = edgeShift(t, net);
    deg = t->deg;

//if(net==3){printtree(*t);getchar();}
    iter = 0;
    cur_pairN1 = cur_pairN2 = -1;
    while(iter<3)
    {
        iter++;
        
        // find all pairs of steiner node at the same position (steiner pairs)
        pairCnt = 0;
        for(i=deg; i<2*deg-2; i++)
        {
            n = t->branch[i].n;
            if(n!=i && n!=t->branch[n].n && t->branch[i].x == t->branch[n].x && t->branch[i].y == t->branch[n].y)
            {
                pairN1[pairCnt] = i;
                pairN2[pairCnt] = n;
                pairCnt ++;
            }
        }
//if(net==3){printf("# pairs: %d, N1=%d, N2=%d\n", pairCnt, pairN1[0], pairN2[0]);getchar();}
        if(pairCnt>0)
        {
            if(pairN1[0]!=cur_pairN1 || pairN2[0]!=cur_pairN2) // don't try the same as last one
            {
                cur_pairN1 = pairN1[0];
                cur_pairN2 = pairN2[0];
                isPair = TRUE;
            }
            else if(pairN1[0]==cur_pairN1 && pairN2[0]==cur_pairN2 && pairCnt>1)
            {
                cur_pairN1 = pairN1[1];
                cur_pairN2 = pairN2[1];
                isPair = TRUE;
            }
            else
                isPair = FALSE;
            
//if(net==3){printf("isPair: %d, N1=%d, N2=%d\n", isPair, cur_pairN1, cur_pairN2);getchar();}
            if(isPair) // find a new pair to swap
            {
                N1nbrH = N1nbrV = N2nbrH = N2nbrV = -1;
                // find the nodes directed to cur_pairN1(2 nodes) and cur_pairN2(1 nodes)
                for(j=0; j<2*deg-2; j++)
                {
                    n = t->branch[j].n;
                    if(n==cur_pairN1)
                    {
                        if(t->branch[j].x==t->branch[cur_pairN1].x && t->branch[j].y!=t->branch[cur_pairN1].y)
                            N1nbrV = j;
                        else if(t->branch[j].y==t->branch[cur_pairN1].y && t->branch[j].x!=t->branch[cur_pairN1].x)
                            N1nbrH = j;
                    }
                    else if(n==cur_pairN2)
                    {
                        if(t->branch[j].x==t->branch[cur_pairN2].x && t->branch[j].y!=t->branch[cur_pairN2].y)
                            N2nbrV = j;
                        else if(t->branch[j].y==t->branch[cur_pairN2].y && t->branch[j].x!=t->branch[cur_pairN2].x)
                            N2nbrH = j;
                    }
                }
                // find the node cur_pairN2 directed to
                n = t->branch[cur_pairN2].n;
                if(t->branch[n].x==t->branch[cur_pairN2].x && t->branch[n].y!=t->branch[cur_pairN2].y)
                    N2nbrV = n;
                else if(t->branch[n].y==t->branch[cur_pairN2].y && t->branch[n].x!=t->branch[cur_pairN2].x)
                    N2nbrH = n;
                
//if(net==3){printf("N1=%d, N2=%d, N1nbrH=%d, N1nbrV=%d, N2nbrH=%d, N2nbrV=%d\n", cur_pairN1, cur_pairN2, N1nbrH, N1nbrV, N2nbrH, N2nbrV);getchar();}
                if(N1nbrH>=0 && N2nbrH>=0)
                {
                    if(N2nbrH==t->branch[cur_pairN2].n)
                    {
                        t->branch[N1nbrH].n = cur_pairN2;
                        t->branch[cur_pairN1].n = N2nbrH;
                        t->branch[cur_pairN2].n = cur_pairN1;
                    }
                    else
                    {
                        t->branch[N1nbrH].n = cur_pairN2;
                        t->branch[N2nbrH].n = cur_pairN1;
                    }
                    numShift += edgeShift(t, net);
                }
                else if(N1nbrV>=0 && N2nbrV>=0)
                {
                    if(N2nbrV==t->branch[cur_pairN2].n)
                    {
                        t->branch[N1nbrV].n = cur_pairN2;
                        t->branch[cur_pairN1].n = N2nbrV;
                        t->branch[cur_pairN2].n = cur_pairN1;
                    }
                    else
                    {
                        t->branch[N1nbrV].n = cur_pairN2;
                        t->branch[N2nbrV].n = cur_pairN1;
                    }
                    numShift += edgeShift(t, net);
                }
//if(net==3){printtree(*t);getchar();}                
             } // if(isPair)
            

        } //if(pairCnt>0)
        else
            iter = 3;

    } // while

    return(numShift);
}
#endif
