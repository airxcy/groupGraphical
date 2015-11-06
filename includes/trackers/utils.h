//
//  utils.h
//  ITF_Inegrated
//
//  Created by ChenYang Xia on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef UTILS
#define UTILS
#include <algorithm>
#include <vector>
class FeatBuff;
int getLineIdx(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB);
int getLineProp(std::vector<int>& x_idx,std::vector<int>&  y_idx,int* PointA,int* PointB,double linedist);
void HSVtoRGB(unsigned char *r, unsigned char *g, unsigned char *b, float h, float s, float v );
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct cvxPnt {
    float x, y;

    bool operator <(const cvxPnt &p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
};
void convex_hull(std::vector<cvxPnt>& P, FeatBuff &H);
void* zalloc(int num,int size);
#endif // UTILS

