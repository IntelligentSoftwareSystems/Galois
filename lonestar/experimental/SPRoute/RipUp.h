#ifndef _RIPUP_H_
#define _RIPUP_H_

#include <stdio.h>
#include <stdlib.h>
#include "DataType.h"
#include "flute.h"
#include "DataProc.h"
#include "route.h"
#include "utility.h"

// rip-up a L segment
void ripupSegL(Segment *seg)
{
    int i, grid;
    int ymin, ymax;
    
    if(seg->y1<seg->y2)
    {
        ymin = seg->y1;
        ymax = seg->y2;
    }
    else
    {
        ymin = seg->y2;
        ymax = seg->y1;
    }
    
    // remove L routing
    if(seg->xFirst)
    {
        grid = seg->y1*(xGrid-1);
        for(i=seg->x1; i<seg->x2; i++)
            h_edges[grid+i].est_usage -= 1;
        for(i=ymin; i<ymax; i++)
            v_edges[i*xGrid+seg->x2].est_usage -= 1;
    }
    else
    {
        for(i=ymin; i<ymax; i++)
            v_edges[i*xGrid+seg->x1].est_usage -= 1;
        grid = seg->y2*(xGrid-1);
        for(i=seg->x1; i<seg->x2; i++)
            h_edges[grid+i].est_usage -= 1;
    }
}

void ripupSegZ(Segment *seg)
{
    int i, grid;
    int ymin , ymax;
    
    if(seg->y1<seg->y2)
    {
        ymin = seg->y1;
        ymax = seg->y2;
    }
    else
    {
        ymin = seg->y2;
        ymax = seg->y1;
    }
    
    if(seg->x1 == seg->x2)
    {
        // remove V routing
        for(i=ymin; i<ymax; i++)
            v_edges[i*xGrid+seg->x1].est_usage -= 1;
    }
    else if(seg->y1 == seg->y2)
    {
        // remove H routing
        grid = seg->y1*(xGrid-1);
        for(i=seg->x1; i<seg->x2; i++)
            h_edges[grid+i].est_usage -= 1;
    }
    else
    {
        // remove Z routing            
        if(seg->HVH)
        {
            grid = seg->y1*(xGrid-1);
            for(i=seg->x1; i<seg->Zpoint; i++)
                h_edges[grid+i].est_usage -= 1;
            grid = seg->y2*(xGrid-1);
            for(i=seg->Zpoint; i<seg->x2; i++)
                h_edges[grid+i].est_usage -= 1;
            for(i=ymin; i<ymax; i++)
                v_edges[i*xGrid+seg->Zpoint].est_usage -= 1;
        }
        else
        {
            if(seg->y1<seg->y2)
            {
                for(i=seg->y1; i<seg->Zpoint; i++)
                    v_edges[i*xGrid+seg->x1].est_usage -= 1;
                for(i=seg->Zpoint; i<seg->y2; i++)
                    v_edges[i*xGrid+seg->x2].est_usage -= 1;
                grid = seg->Zpoint*(xGrid-1);
                for(i=seg->x1; i<seg->x2; i++)
                    h_edges[grid+i].est_usage -= 1;
            }
            else
            {
                for(i=seg->y2; i<seg->Zpoint; i++)
                    v_edges[i*xGrid+seg->x2].est_usage -= 1;
                for(i=seg->Zpoint; i<seg->y1; i++)
                    v_edges[i*xGrid+seg->x1].est_usage -= 1;
                grid = seg->Zpoint*(xGrid-1);
                for(i=seg->x1; i<seg->x2; i++)
                    h_edges[grid+i].est_usage -= 1;
            }
        }
    }

}

void newRipup(TreeEdge* treeedge, TreeNode *treenodes, int x1, int y1, int x2, int y2)
{
	short *gridsX, *gridsY;
    int i, j, grid, Zpoint, ymin, ymax, xmin, n1, n2;
    RouteType ripuptype;

    if(treeedge->len==0) 
	{
		return ; // not ripup for degraded edge
	}

	n1 = treeedge->n1;
	n2 = treeedge->n2;


        
    ripuptype = treeedge->route.type;
    if(y1<y2)
    {
        ymin = y1;
        ymax = y2;
    }
    else
    {
        ymin = y2;
        ymax = y1;
    }
    
    if(ripuptype==LROUTE) // remove L routing
    {
        if(treeedge->route.xFirst)
        {
			grid = y1*(xGrid-1);
            for(i=x1; i<x2; i++)
                h_edges[grid+i].est_usage -= 1;
            for(i=ymin; i<ymax; i++)
                v_edges[i*xGrid+x2].est_usage -= 1;
        }
        else
        {
            for(i=ymin; i<ymax; i++)
                v_edges[i*xGrid+x1].est_usage -= 1;
            grid = y2*(xGrid-1);
            for(i=x1; i<x2; i++)
                h_edges[grid+i].est_usage -= 1;
        }
    }
    else if(ripuptype==ZROUTE)
    {
        // remove Z routing            
        Zpoint = treeedge->route.Zpoint;
        if(treeedge->route.HVH)
        {
            grid = y1*(xGrid-1);
            for(i=x1; i<Zpoint; i++)
                h_edges[grid+i].est_usage -= 1;
            grid = y2*(xGrid-1);
            for(i=Zpoint; i<x2; i++)
                h_edges[grid+i].est_usage -= 1;
            for(i=ymin; i<ymax; i++)
                v_edges[i*xGrid+Zpoint].est_usage -= 1;
        }
        else
        {
            if(y1<y2)
            {
                for(i=y1; i<Zpoint; i++)
                    v_edges[i*xGrid+x1].est_usage -= 1;
                for(i=Zpoint; i<y2; i++)
                    v_edges[i*xGrid+x2].est_usage -= 1;
                grid = Zpoint*(xGrid-1);
                for(i=x1; i<x2; i++)
                    h_edges[grid+i].est_usage -= 1;
            }
            else
            {
                for(i=y2; i<Zpoint; i++)
                    v_edges[i*xGrid+x2].est_usage -= 1;
                for(i=Zpoint; i<y1; i++)
                    v_edges[i*xGrid+x1].est_usage -= 1;
                grid = Zpoint*(xGrid-1);
                for(i=x1; i<x2; i++)
                    h_edges[grid+i].est_usage -= 1;
            }
        }
    }
    else if(ripuptype==MAZEROUTE)
    {
        gridsX = treeedge->route.gridsX;
        gridsY = treeedge->route.gridsY;
        for(i=0; i<treeedge->route.routelen; i++)
        {
            if(gridsX[i]==gridsX[i+1]) // a vertical edge
            {
                ymin = min(gridsY[i], gridsY[i+1]);
                v_edges[ymin*xGrid+gridsX[i]].est_usage -= 1;
            }
            else if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
                xmin = min(gridsX[i], gridsX[i+1]);
                h_edges[gridsY[i]*(xGrid-1)+xmin].est_usage -= 1;
            }
            else
            {
                printf("MAZE RIPUP WRONG\n");
				for(j=0; j<treeedge->route.routelen; j++)
				{
					printf("x %d y %d\n",gridsX[j],gridsY[j]);
				}
				exit(1);
            }
        }
    }
}

Bool newRipupType2(TreeEdge* treeedge, TreeNode *treenodes, int x1, int y1, int x2, int y2, int deg)
{
    int i, j, grid, Zpoint, ymin, ymax, xmin, n1, n2;
    int *gridsX, *gridsY;
    RouteType ripuptype;
	Bool needRipup = FALSE;

	if(treeedge->len==0) {
		return (FALSE); // not ripup for degraded edge
	}
        
    ripuptype = treeedge->route.type;
    if(y1<y2)
    {
        ymin = y1;
        ymax = y2;
    }
    else
    {
        ymin = y2;
        ymax = y1;
    }
    
    if(ripuptype==LROUTE) // remove L routing
    {
        if(treeedge->route.xFirst)
        {
            grid = y1*(xGrid-1);
			for(i=x1; i<x2; i++) {
				if (h_edges[grid+i].est_usage > h_edges[grid+i].cap) {
					needRipup = TRUE;
                    break;
				}
			}

			for(i=ymin; i<ymax; i++) {
				if (v_edges[i*xGrid+x2].est_usage > v_edges[i*xGrid+x2].cap ) {
					needRipup = TRUE;
                    break;
				}
			}
        }
        else
        {
			for(i=ymin; i<ymax; i++) {
				if (v_edges[i*xGrid+x1].est_usage > v_edges[i*xGrid+x1].cap) {
					needRipup = TRUE;
                    break;
				}
			}
            grid = y2*(xGrid-1);
			for(i=x1; i<x2; i++) {
				if ( h_edges[grid+i].est_usage >  v_edges[grid+i].cap) {
					needRipup = TRUE;
                    break;
				}
			}
        }

		if (needRipup) {
			n1 = treeedge->n1;
			n2 = treeedge->n2;
			
			if(treeedge->route.xFirst)
			{
				if (n1 >= deg) {
					treenodes[n1].status -=2;
				}
				treenodes[n2].status -=1;

				grid = y1*(xGrid-1);
				for(i=x1; i<x2; i++)
					h_edges[grid+i].est_usage -= 1;
				for(i=ymin; i<ymax; i++)
					v_edges[i*xGrid+x2].est_usage -= 1;
			}
			else
			{
				if (n2 >= deg) {
					treenodes[n2].status -=2;
				}
				treenodes[n1].status -=1;

				for(i=ymin; i<ymax; i++)
					v_edges[i*xGrid+x1].est_usage -= 1;
				grid = y2*(xGrid-1);
				for(i=x1; i<x2; i++)
					h_edges[grid+i].est_usage -= 1;
			}
		}
		return (needRipup);

	} else  {
		printf("type2 ripup not type L\n");
		exit(0);
	}
}


void printEdgeVEC(TreeEdge* treeedge)
{
    int i;
    
    for(i=0; i<=treeedge->route.routelen; i++)
    {
        printf("(%d, %d) ", treeedge->route.gridsX[i], treeedge->route.gridsY[i]);
    }
    printf("\n");
}

Bool newRipupCheckProb(TreeEdge* treeedge, int x1, int y1, int x2, int y2,int ripup_threshold, int netID, int edgeID)
{
	short *gridsX, *gridsY;
    int i, grid, Zpoint, ymin, xmin, max_usageH, max_usageV;
    Bool needRipup = FALSE;
    
	if(treeedge->len==0) {
		return (FALSE);
	}// not ripup for degraded edge

    //std::random_device rd;
    //std::mt19937 g(rd());

    if(treeedge->route.type==MAZEROUTE)
    {
        gridsX = treeedge->route.gridsX;
        gridsY = treeedge->route.gridsY;
        for(i=0; i<treeedge->route.routelen; i++)
        {
            if(gridsX[i]==gridsX[i+1]) // a vertical edge
            {
                ymin = min(gridsY[i], gridsY[i+1]);
				grid = ymin*xGrid+gridsX[i];
				int cap = vCapacity - ripup_threshold - v_edges[grid].red;
				int overflow = v_edges[grid].usage +v_edges[grid].red - vCapacity - ripup_threshold;
                int r = rand(); 
                //if(overflow >= 0) printf("red %d r %d cap %d %d overflow %d ripup: %d\n", v_edges[grid].red, r, cap, r%cap, overflow, (int)(r%cap <= overflow));
                if(overflow >= 0 && (r%cap <= overflow) )
                {
                    needRipup = TRUE;
                    break;
                }
            }
            else if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
                xmin = min(gridsX[i], gridsX[i+1]);
				grid = gridsY[i]*(xGrid-1)+xmin;
				int cap = hCapacity - ripup_threshold - h_edges[grid].red;
				int overflow = h_edges[grid].usage + h_edges[grid].red - hCapacity - ripup_threshold;
                int r = rand();
                //if(overflow >= 0) printf("red %d r %d cap %d %d overflow %d ripup: %d\n", h_edges[grid].red, r, cap, r%cap, overflow, (int)(r%cap <= overflow));
                if(overflow >= 0 && (r%cap <= overflow) )
                {
                    needRipup = TRUE;
                    break;
                }
            }
        }

        
        if(needRipup)
        {
        	
            for(i=0; i<treeedge->route.routelen; i++)
            {
                if(gridsX[i]==gridsX[i+1]) // a vertical edge
                {
                    ymin = min(gridsY[i], gridsY[i+1]);
                    //v_edges[ymin*xGrid+gridsX[i]].usage -= 1;
                    v_edges[ymin*xGrid+gridsX[i]].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
                else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                {
                    xmin = min(gridsX[i], gridsX[i+1]);
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage -= 1;
                    h_edges[gridsY[i]*(xGrid-1)+xmin].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
            }
            
            return(TRUE);
        }
		else {
            return(FALSE);
		}
    } else {
        printf("route type is not maze, netID %d\n",netID);
		fflush(stdout);
		printEdge(netID, edgeID);
		
		exit(0);
    }
    

}

Bool newRipupCheck(TreeEdge* treeedge, int x1, int y1, int x2, int y2,int ripup_threshold, int netID, int edgeID)
{
	short *gridsX, *gridsY;
    int i, grid, Zpoint, ymin, xmin, max_usageH, max_usageV;
    Bool needRipup = FALSE;
    
	if(treeedge->len==0) {
		return (FALSE);
	}// not ripup for degraded edge
        
    if(treeedge->route.type==MAZEROUTE)
    {
        gridsX = treeedge->route.gridsX;
        gridsY = treeedge->route.gridsY;
        for(i=0; i<treeedge->route.routelen; i++)
        {
            if(gridsX[i]==gridsX[i+1]) // a vertical edge
            {
                ymin = min(gridsY[i], gridsY[i+1]);
				grid = ymin*xGrid+gridsX[i];
                if(v_edges[grid].usage +v_edges[grid].red >= vCapacity - ripup_threshold)
                {
                    needRipup = TRUE;
                    break;
                }

            }
            else if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
                xmin = min(gridsX[i], gridsX[i+1]);
				grid = gridsY[i]*(xGrid-1)+xmin;
                if(h_edges[grid].usage + h_edges[grid].red >= hCapacity - ripup_threshold)
                {
                    needRipup = TRUE;
                    break;
                }
            }
        }

        
        if(needRipup)
        {
            for(i=0; i<treeedge->route.routelen; i++)
            {
                if(gridsX[i]==gridsX[i+1]) // a vertical edge
                {
                    ymin = min(gridsY[i], gridsY[i+1]);
                    //v_edges[ymin*xGrid+gridsX[i]].usage -= 1;
                    v_edges[ymin*xGrid+gridsX[i]].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
                else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                {
                    xmin = min(gridsX[i], gridsX[i+1]);
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage -= 1;
                    h_edges[gridsY[i]*(xGrid-1)+xmin].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
            }
            
            return(TRUE);
        }
		else {
            return(FALSE);
		}
    } else {
        printf("route type is not maze, netID %d\n",netID);
		fflush(stdout);
		printEdge(netID, edgeID);
		
		exit(0);
    }
}

Bool newRipupCheck_M1M2(TreeEdge* treeedge, int x1, int y1, int x2, int y2,int ripup_threshold, int netID, int edgeID)
{
    short *gridsX, *gridsY;
    int i, grid, Zpoint, ymin, xmin, max_usageH, max_usageV;
    Bool needRipup = FALSE;
    
    if(treeedge->len==0) {
        return (FALSE);
    }// not ripup for degraded edge
        
    
    if(treeedge->route.type==MAZEROUTE)
    {
        gridsX = treeedge->route.gridsX;
        gridsY = treeedge->route.gridsY;
        for(i=0; i<treeedge->route.routelen; i++)
        {
            if(gridsX[i]==gridsX[i+1]) // a vertical edge
            {
                ymin = min(gridsY[i], gridsY[i+1]);
                grid = ymin*xGrid+gridsX[i];
                if(v_edges[grid].usage +v_edges[grid].red >= vCapacity - ripup_threshold && treeedge->n_ripups <= v_edges[grid].max_have_rippedups)
                {
                    v_edges[grid].ripups_cur_round = true;
                    needRipup = TRUE;
                    break;
                }

            }
            else if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
                xmin = min(gridsX[i], gridsX[i+1]);
                grid = gridsY[i]*(xGrid-1)+xmin;
                if(h_edges[grid].usage + h_edges[grid].red >= hCapacity - ripup_threshold && treeedge->n_ripups <= h_edges[grid].max_have_rippedups)
                {
                    h_edges[grid].ripups_cur_round = true;
                    needRipup = TRUE;
                    break;
                }
            }
        }

        
        if(needRipup)
        {
            for(i=0; i<treeedge->route.routelen; i++)
            {
                if(gridsX[i]==gridsX[i+1]) // a vertical edge
                {
                    ymin = min(gridsY[i], gridsY[i+1]);
                    //v_edges[ymin*xGrid+gridsX[i]].usage -= 1;
                    v_edges[ymin*xGrid+gridsX[i]].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
                else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                {
                    xmin = min(gridsX[i], gridsX[i+1]);
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage -= 1;
                    h_edges[gridsY[i]*(xGrid-1)+xmin].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
            }
            
            return(TRUE);
        }
        else {
            return(FALSE);
        }
    } else {
        printf("route type is not maze, netID %d\n",netID);
        fflush(stdout);
        printEdge(netID, edgeID);
        
        exit(0);
    }
}


Bool newRipupCheck_atomic(TreeEdge* treeedge, int x1, int y1, int x2, int y2,int ripup_threshold, int netID, int edgeID)
{
    short *gridsX, *gridsY;
    int i, grid, Zpoint, ymin, xmin, max_usageH, max_usageV;
    Bool needRipup = FALSE;
    bool break_edge_loop = false;
    int break_edge = 0;
    
    if(treeedge->len==0) {
        return (FALSE);
    }// not ripup for degraded edge
    //std::cout << " atomic ripup" << std::endl;
    if(treeedge->route.type==MAZEROUTE)
    {
        gridsX = treeedge->route.gridsX;
        gridsY = treeedge->route.gridsY;
        for(i=0; i<treeedge->route.routelen; i++)
        {
            if(gridsX[i]==gridsX[i+1]) // a vertical edge
            {
                ymin = min(gridsY[i], gridsY[i+1]);
                grid = ymin*xGrid+gridsX[i];
                int old_usage = v_edges[grid].usage;

                while(old_usage + v_edges[grid].red >= vCapacity - ripup_threshold)
                {
                    if(v_edges[grid].usage.compare_exchange_weak(old_usage, old_usage - 1)) {
                        break_edge = i;                       
                        needRipup = TRUE;
                        break;
                    }
                    old_usage = v_edges[grid].usage;
                }
                if(needRipup)
                    break;

            }
            else if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
                xmin = min(gridsX[i], gridsX[i+1]);
                grid = gridsY[i]*(xGrid-1)+xmin;
                int old_usage = h_edges[grid].usage;

                while(old_usage + h_edges[grid].red >= hCapacity - ripup_threshold)
                {
                    if(h_edges[grid].usage.compare_exchange_weak(old_usage, old_usage - 1)) {
                        break_edge = i;
                        needRipup = TRUE;
                        break;
                    }
                    old_usage = h_edges[grid].usage;
                }
                if(needRipup)
                    break;
            }
        }

        
        if(needRipup)
        {
            for(i=0; i<treeedge->route.routelen; i++)
            {
                if(i == break_edge) continue;
                if(gridsX[i]==gridsX[i+1]) // a vertical edge
                {
                    ymin = min(gridsY[i], gridsY[i+1]);
                    //v_edges[ymin*xGrid+gridsX[i]].usage -= 1;
                    v_edges[ymin*xGrid+gridsX[i]].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
                else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                {
                    xmin = min(gridsX[i], gridsX[i+1]);
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage -= 1;
                    h_edges[gridsY[i]*(xGrid-1)+xmin].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
            }
            
            return(TRUE);
        }
        else {
            return(FALSE);
        }
    } else {
        printf("route type is not maze, netID %d\n",netID);
        fflush(stdout);
        printEdge(netID, edgeID);
        
        exit(0);
    }
}

Bool newRipupCheck_sort(TreeEdge* treeedge, int x1, int y1, int x2, int y2,int ripup_threshold, int netID, int edgeID, 
	bool& is_horizontal, int& grid_pos)
{
	short *gridsX, *gridsY;
    int i, grid, Zpoint, ymin, xmin, max_usageH, max_usageV;
    Bool needRipup = FALSE;
    treeedge->ripup = false;
	if(treeedge->len==0) {
		return (FALSE);
	}// not ripup for degraded edge
        
    if(treeedge->route.type==MAZEROUTE)
    {
        gridsX = treeedge->route.gridsX;
        gridsY = treeedge->route.gridsY;
        for(i=0; i<treeedge->route.routelen; i++)
        {
            if(gridsX[i]==gridsX[i+1]) // a vertical edge
            {
                ymin = min(gridsY[i], gridsY[i+1]);
				grid = ymin*xGrid+gridsX[i];
                if(v_edges[grid].usage +v_edges[grid].red >= vCapacity - ripup_threshold)
                {
                    needRipup = TRUE;
                    is_horizontal = false;
                    grid_pos = grid;
                    break;
                }

            }
            else if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
                xmin = min(gridsX[i], gridsX[i+1]);
				grid = gridsY[i]*(xGrid-1)+xmin;
                if(h_edges[grid].usage + h_edges[grid].red >= hCapacity - ripup_threshold)
                {
                    needRipup = TRUE;
                    is_horizontal = true;
                    grid_pos = grid;
                    break;
                }
            }
        }

        
        if(needRipup)
        {
            /*for(i=0; i<treeedge->route.routelen; i++)
            {
                if(gridsX[i]==gridsX[i+1]) // a vertical edge
                {
                    ymin = min(gridsY[i], gridsY[i+1]);
                    if(netID == 2 && edgeID == 21)
                        printf("i %d x %d y %d\n", i, gridsX[i], gridsY[i]);
                    //v_edges[ymin*xGrid+gridsX[i]].usage -= 1;
                    //v_edges[ymin*xGrid+gridsX[i]].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
                else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                {
                    xmin = min(gridsX[i], gridsX[i+1]);
                    if(netID == 2 && edgeID == 21)
                        printf("i %d x %d y %d\n", i, gridsX[i], gridsY[i]);
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage -= 1;
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
            }*/
            
            return(TRUE);
        }
		else {
            return(FALSE);
		}
    } else {
        printf("route type is not maze, netID %d\n",netID);
		fflush(stdout);
		printEdge(netID, edgeID);
		
		exit(0);
    }
    

}

Bool newRipupCheck_nosub(TreeEdge* treeedge, int x1, int y1, int x2, int y2,int ripup_threshold, int netID, int edgeID)
{
	short *gridsX, *gridsY;
    int i, grid, Zpoint, ymin, xmin, max_usageH, max_usageV;
    Bool needRipup = FALSE;
    
	if(treeedge->len==0) {
		return (FALSE);
	}// not ripup for degraded edge
        
    if(treeedge->route.type==MAZEROUTE)
    {
        gridsX = treeedge->route.gridsX;
        gridsY = treeedge->route.gridsY;
        for(i=0; i<treeedge->route.routelen; i++)
        {
            if(gridsX[i]==gridsX[i+1]) // a vertical edge
            {
                ymin = min(gridsY[i], gridsY[i+1]);
                grid = ymin*xGrid+gridsX[i];
                if(v_edges[grid].usage +v_edges[grid].red >= vCapacity - ripup_threshold)
                {
                    needRipup = TRUE;
                    break;
                }

            }
            else if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
                xmin = min(gridsX[i], gridsX[i+1]);
                grid = gridsY[i]*(xGrid-1)+xmin;
                if(h_edges[grid].usage + h_edges[grid].red >= hCapacity - ripup_threshold)
                {
                    needRipup = TRUE;
                    break;
                }
            }
        }

        
        if(needRipup)
        {
            /*for(i=0; i<treeedge->route.routelen; i++)
            {
                if(pre_gridsX[i]==pre_gridsX[i+1]) // a vertical edge
                {
                    ymin = min(pre_gridsY[i], pre_gridsY[i+1]);
                    printf("nosub x y %d %d i %d\n", pre_gridsX[i], ymin, i);
                    //v_edges[ymin*xGrid+gridsX[i]].usage -= 1;
                    //v_edges[ymin*xGrid+gridsX[i]].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
                else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                {
                    xmin = min(pre_gridsX[i], pre_gridsX[i+1]);
                    printf("nosub x y %d %d i %d \n", xmin, pre_gridsY[i], i);
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage -= 1;
                    //h_edges[gridsY[i]*(xGrid-1)+xmin].usage.fetch_sub((short unsigned)1, std::memory_order_relaxed);
                }
            }*/
            
            return(TRUE);
        }
		else {
            return(FALSE);
		}
    } else {
        printf("route type is not maze, netID %d\n",netID);
		fflush(stdout);
		printEdge(netID, edgeID);
		
		exit(0);
    }
    

}



Bool newRipup3DType3(int netID, int edgeID)
{
	short *gridsX, *gridsY, *gridsL;
    int i, k, grid, Zpoint, ymin, ymax, xmin, lv, lh, n1a, n2a, hl, bl, hid, bid, deg;
   
    RouteType ripuptype;
	TreeEdge *treeedges, *treeedge;
    TreeNode *treenodes;

	treeedges = sttrees[netID].edges;
	treeedge = &(treeedges[edgeID]);

	if(treeedge->len==0) {
		return (FALSE); // not ripup for degraded edge
	}

	treenodes = sttrees[netID].nodes;

	deg =  sttrees[netID].deg;

	n1a = treeedge->n1a;
	n2a = treeedge->n2a;

	if (n1a < deg) {
		bl = 0;
	} else {
		bl = BIG_INT;
	}
	hl = 0;
	hid = bid = BIG_INT;

	for (i = 0; i < treenodes[n1a].conCNT; i ++) {
		if (treenodes[n1a].eID[i] == edgeID) {
			for (k = i+1; k < treenodes[n1a].conCNT; k++) {
				treenodes[n1a].eID[k-1] = treenodes[n1a].eID[k];
				treenodes[n1a].heights[k-1] = treenodes[n1a].heights[k] ;
				if (bl > treenodes[n1a].heights[k]) {
					bl = treenodes[n1a].heights[k];
					bid = treenodes[n1a].eID[k];
				}
				if (hl < treenodes[n1a].heights[k]) {
					hl = treenodes[n1a].heights[k];
					hid = treenodes[n1a].eID[k];
				}
			}
			break;
		} else {
			if (bl > treenodes[n1a].heights[i]) {
				bl = treenodes[n1a].heights[i];
				bid = treenodes[n1a].eID[i];
			}
			if (hl < treenodes[n1a].heights[i]) {
				hl = treenodes[n1a].heights[i];
				hid = treenodes[n1a].eID[i];
			}
		}
	}
	treenodes[n1a].conCNT --;

	treenodes[n1a].botL = bl;
	treenodes[n1a].lID = bid;
	treenodes[n1a].topL = hl;
	treenodes[n1a].hID = hid;

	if (n2a < deg) {
		bl = 0;
	} else {
		bl = BIG_INT;
	}
	hl = 0;
	hid = bid = BIG_INT;


	for (i = 0; i < treenodes[n2a].conCNT; i ++) {
		if (treenodes[n2a].eID[i] == edgeID) {
			for (k = i+1; k < treenodes[n2a].conCNT; k++) {
				treenodes[n2a].eID[k-1] = treenodes[n2a].eID[k];
				treenodes[n2a].heights[k-1] = treenodes[n2a].heights[k] ;
				if (bl > treenodes[n2a].heights[k]) {
					bl = treenodes[n2a].heights[k];
					bid = treenodes[n2a].eID[k];
				}
				if (hl < treenodes[n2a].heights[k]) {
					hl = treenodes[n2a].heights[k];
					hid = treenodes[n2a].eID[k];
				}
			}
			break;
		} else {
			if (bl > treenodes[n2a].heights[i]) {
				bl = treenodes[n2a].heights[i];
				bid = treenodes[n2a].eID[i];
			}
			if (hl < treenodes[n2a].heights[i]) {
				hl = treenodes[n2a].heights[i];
				hid = treenodes[n2a].eID[i];
			}
		}
	}
	treenodes[n2a].conCNT --;

	treenodes[n2a].botL = bl;
	treenodes[n2a].lID = bid;
	treenodes[n2a].topL = hl;
	treenodes[n2a].hID = hid;

	gridsX = treeedge->route.gridsX;
	gridsY = treeedge->route.gridsY;
	gridsL = treeedge->route.gridsL;
	for(i=0; i<treeedge->route.routelen; i++)
	{
		if (gridsL[i] ==  gridsL[i+1]) {
			if(gridsX[i]==gridsX[i+1]) // a vertical edge
			{
				ymin = min(gridsY[i], gridsY[i+1]);
				grid = gridsL[i]*gridV+ymin*xGrid+gridsX[i];
				v_edges3D[grid].usage -= 1;
			}
			else if(gridsY[i]==gridsY[i+1])// a horizontal edge
			{
				xmin = min(gridsX[i], gridsX[i+1]);
				grid = gridsL[i]*gridH+gridsY[i]*(xGrid-1)+xmin;
				h_edges3D[grid].usage -= 1;
			} else {
				printf("MAZE RIPUP WRONG\n");
				return (FALSE);
				//exit(1);
			}
		}
	}

	return (TRUE);    

}

void newRipupNet(int netID)
{
	short *gridsX, *gridsY;
    int i, j, grid, Zpoint, ymin, ymax, xmin, n1, n2, edgeID;
    
    RouteType ripuptype;
	TreeEdge *treeedges, *treeedge;
	TreeNode *treenodes;
	int x1, y1, x2, y2, deg;

	treeedges = sttrees[netID].edges;
	treenodes=sttrees[netID].nodes;
	deg = sttrees[netID].deg;

	for (edgeID = 0; edgeID < 2*deg-3; edgeID++) {
		treeedge = &(treeedges[edgeID]);
		if (treeedge->len > 0) {

			n1 = treeedge->n1;
			n2 = treeedge->n2;
			x1 = treenodes[n1].x;
            y1 = treenodes[n1].y;
            x2 = treenodes[n2].x;
            y2 = treenodes[n2].y;

			ripuptype = treeedge->route.type;
			if(y1<y2)
			{
				ymin = y1;
				ymax = y2;
			}
			else
			{
				ymin = y2;
				ymax = y1;
			}
		    
			if(ripuptype==LROUTE) // remove L routing
			{
				if(treeedge->route.xFirst)
				{
					grid = y1*(xGrid-1);
					for(i=x1; i<x2; i++)
						h_edges[grid+i].est_usage -= 1;
					for(i=ymin; i<ymax; i++)
						v_edges[i*xGrid+x2].est_usage -= 1;
				}
				else
				{
					for(i=ymin; i<ymax; i++)
						v_edges[i*xGrid+x1].est_usage -= 1;
					grid = y2*(xGrid-1);
					for(i=x1; i<x2; i++)
						h_edges[grid+i].est_usage -= 1;
				}
			}
			else if(ripuptype==ZROUTE)
			{
				// remove Z routing            
				Zpoint = treeedge->route.Zpoint;
				if(treeedge->route.HVH)
				{
					grid = y1*(xGrid-1);
					for(i=x1; i<Zpoint; i++)
						h_edges[grid+i].est_usage -= 1;
					grid = y2*(xGrid-1);
					for(i=Zpoint; i<x2; i++)
						h_edges[grid+i].est_usage -= 1;
					for(i=ymin; i<ymax; i++)
						v_edges[i*xGrid+Zpoint].est_usage -= 1;
				}
				else
				{
					if(y1<y2)
					{
						for(i=y1; i<Zpoint; i++)
							v_edges[i*xGrid+x1].est_usage -= 1;
						for(i=Zpoint; i<y2; i++)
							v_edges[i*xGrid+x2].est_usage -= 1;
						grid = Zpoint*(xGrid-1);
						for(i=x1; i<x2; i++)
							h_edges[grid+i].est_usage -= 1;
					}
					else
					{
						for(i=y2; i<Zpoint; i++)
							v_edges[i*xGrid+x2].est_usage -= 1;
						for(i=Zpoint; i<y1; i++)
							v_edges[i*xGrid+x1].est_usage -= 1;
						grid = Zpoint*(xGrid-1);
						for(i=x1; i<x2; i++)
							h_edges[grid+i].est_usage -= 1;
					}
				}
			}
			else if(ripuptype==MAZEROUTE)
			{
				gridsX = treeedge->route.gridsX;
				gridsY = treeedge->route.gridsY;
				for(i=0; i<treeedge->route.routelen; i++)
				{
					if(gridsX[i]==gridsX[i+1]) // a vertical edge
					{
						ymin = min(gridsY[i], gridsY[i+1]);
						v_edges[ymin*xGrid+gridsX[i]].est_usage -= 1;
					}
					else if(gridsY[i]==gridsY[i+1])// a horizontal edge
					{
						xmin = min(gridsX[i], gridsX[i+1]);
						h_edges[gridsY[i]*(xGrid-1)+xmin].est_usage -= 1;
					}
					else
					{
						printf("MAZE RIPUP WRONG in newRipupNet\n");
						for(j=0; j<treeedge->route.routelen; j++)
						{
							printf("x %d y %d\n",gridsX[j],gridsY[j]);
                            //if(gridsX[i]!=gridsX[i+1] && gridsY[i]==gridsY[i+1])
						}
						//exit(1);
					}
				}
			}
		}
    }
}

#endif

