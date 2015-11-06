#include "trackers/utils.h"
#include "trackers/buffers.h"
#include <iostream>


// 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
// Returns a positive value, if OAB makes a counter-clockwise turn,
// negative for clockwise turn, and zero if the points are collinear.
float cross(const FeatPts_p O, const FeatPts_p A, const cvxPnt &B)
{
    return (A->x - O->x) * (B.y - O->y) - (A->y - O->y) * (B.x - O->x);
}
void* zalloc(int num,int size)
{
    int bsize = size*num;
    void* ptr=malloc(bsize);
    memset(ptr,0,bsize);
    return ptr;
}
// Returns a list of points on the convex hull in counter-clockwise order.
// Note: the last point in the returned list is the same as the first one.
void convex_hull(std::vector<cvxPnt>& P,FeatBuff& H)
{
    int n = P.size(), k = 0;
    H.clear();
    // Sort points lexicographically
    sort(P.begin(), P.end());
    FeatPts_p ptptr;
    // Build lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(H.getPtr(k-2), H.getPtr(k-1), P[i]) <= 0) k--;
        ptptr =H.getPtr(k++);
        ptptr->x=P[i].x;
        ptptr->y=P[i].y;
        //H[k++] = P[i];
    }

    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross(H.getPtr(k-2), H.getPtr(k-1), P[i]) <= 0) k--;
        ptptr =H.getPtr(k++);
        ptptr->x=P[i].x;
        ptptr->y=P[i].y;
        //H[k++] = P[i];
    }
    H.len=k;
}
int getLineIdx(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB)
{
    /*
    draw a line from PointA to PointB store postions in x_idx y_idx
    return length of the line always >0
    reference Bresenham's line algorithm
    you need to free memory of x_idx y_idx your self
    */
    int idx_N=0,startx=PointA[0],starty=PointA[1],endx=PointB[0],endy=PointB[1];
    int diffx=(endx-startx),diffy=(endy-starty);
    int dx=(diffx > 0) - (diffx < 0), dy=(diffy > 0) - (diffy < 0);
        diffx=diffx*dx,diffy=diffy*dy;
    int x=startx,y=starty;
    int step,incre,err,thresherr;
    int *increter,*steper;
    if(diffx>=diffy)
    {
        err=diffy;
        thresherr=diffx/2;
        increter=&y;
        incre=dy;
        steper=&x;
        step=dx;
        idx_N=diffx;
    }
    else
    {
        err=diffx;
        thresherr=diffy/2;
        increter=&x;
        incre=dx;
        steper=&y;
        step=dy;
        idx_N=diffy;
    }
    int toterr=0,i=0;
    for(i=0;i<idx_N;i++)
    {
        x_idx.push_back(x);
        y_idx.push_back(y);
        (*steper)+=step;
        toterr+=err;
        if((toterr)>=thresherr)
        {
            toterr=toterr-idx_N;
            (*increter)+=incre;
        }
    }
    x_idx.push_back(PointB[0]);
    y_idx.push_back(PointB[1]);
    return idx_N+1;
}
void HSVtoRGB( unsigned char *r, unsigned char *g, unsigned char *b, float h, float s, float v  )
{
    int i;
    float f;
    int p, q, t,vc=v*255;
    vc=vc*(vc>0);
    int indv=vc<255;
    vc = vc*indv+255*(1-indv);
    if( s <= 0.0 ) {
        *r = *g = *b = vc;
        return;
    }
    h /= 60.0;			// sector 0 to 5
    i =  h ;
    f = h - i;			// factorial part of h
    p = v * ( 1.0 - s )*255;
    q = v * ( 1.0 - s * f )*255;
    t = v * ( 1.0 - s * ( 1.0 - f ) )*255;
    p = p*(p>0),q = q*(q>0),t = t*(t>0);
    int indp=p<255,indq=q<255,indt=t<255;
    p = p*indp+255*(1-indp);
    q = q*indq+255*(1-indq);
    t = t*indt+255*(1-indt);
    switch( i ) {
        case 0:
            *r = vc;
            *g = t;
            *b = p;
            break;
        case 1:
            *r = q;
            *g = vc;
            *b = p;
            break;
        case 2:
            *r = p;
            *g = vc;
            *b = t;
            break;
        case 3:
            *r = p;
            *g = q;
            *b = vc;
            break;
        case 4:
            *r = t;
            *g = p;
            *b = vc;
            break;
    case 5:
    default:
            *r = vc;
            *g = p;
            *b = q;
            break;
    }
}
int getLineProp(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB,double linedist)
{
    int xA=PointA[0],yA=PointA[1],xB=PointB[0],yB=PointB[1];
    int linelen = linedist+0.5+1;
    double xstep=(xB-xA)/linedist;
    double ystep=(yB-yA)/linedist;
    for(int i=0;i<linelen;i++)
    {
        int x = xA+i*xstep+0.5;
        int y = yA+i*ystep+0.5;
        x_idx.push_back(x);
        y_idx.push_back(y);
    }
    return linelen;
}
