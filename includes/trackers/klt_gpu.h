//
//  klt_gpu.h
//  ITF_Inegrated
//
//  Created by Chenyang Xia on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include <string>
#include <vector>
#include "trackers/buffers.h"
#include "trackers/utils.h"
#include <opencv2/core/core.hpp>
class KLTsparse_CUDA
{
public:
    int frame_width=0, frame_height=0;
    int frameidx=0;
    int nFeatures=0,nSearch=0; /*** get frature number ***/
    std::vector<FeatBuff> trackBuff;
    FeatBuff bbTrkBUff;
    FeatPts pttmp;
    ofv ofvtmp;
    Buff<ofv> ofvBuff;
    cvxPnt cvxPnttmp;
    cv::Mat corners;
    /** cuda **/

    //Tracking
    float* h_curvec,* h_persMap;
    bool persDone=false,render=true,applyseg=false,groupOnFlag=true;
    unsigned char* h_roimask;
    void setUpPersMap(float *srcMap);
    void updateSegCPU(unsigned char* ptr);
    void updateSegNeg(float* aryPtr,int length);
    void updateROICPU(float* aryPtr,int length);
    bool checkTrackMoving(FeatBuff &strk);
    void PointTracking();
    void findPoints();
    void filterTrack();
    //Grouping
    int offsetidx=0;
    unsigned char* h_neighbor,* h_clrvec;
    int h_pairN;
    int *h_prelabel,*h_label,*label_final,*h_gcount,*h_overlap,*h_KnnIdx;
    float *h_com,*h_distmat, *h_group;
    int curTrkingIdx=0;
    int h_newcount=0,curK=0,pregroupN=0,groupN=0,maxgroupN=0;
    bool updateFlag=false,calPolyGon=false;
    std::vector<int> items;
    std::vector< std::vector<cvxPnt> > setPts;
    std::vector<FeatBuff> cvxPts;
    void bfsearch();
    void reGroup();

    KLTsparse_CUDA();
    ~KLTsparse_CUDA();
    int init(int w,int h,unsigned char* framedata,int nPoints);
	int selfinit(unsigned char* framedata);
    int updateAframe(unsigned char* framedata,int fidx);


	int endTraking();
};
#endif
