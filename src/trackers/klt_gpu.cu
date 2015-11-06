#include "trackers/klt_gpu.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "trackers/targetfinder.h"
using namespace cv;
using namespace std;
#define PI 3.14159265


#define minGSize 1
#define MoveFactor 0.00001
#define updateRate 0.1
#define CoThresh 0//(5.0/1000000)
#define CoNb 0.6
#define KnnK 40
#define minTrkLen 2

#define gpu_zalloc(ptr, num, size) cudaMalloc(&ptr,size*num);cudaMemset(ptr,0,size*num);

Mat prePts,nextPts,status,eigenvec;

//cv::gpu::GoodFeaturesToTrackDetector_GPU* detector;
TargetFinder* detector;
cv::gpu::PyrLKOpticalFlow* tracker;
cv::gpu::GpuMat gpuGray, gpuPreGray,rgbMat,maskMat,roiMaskMat,gpuPersMap,gpuSegMat,
        gpuCorners, gpuPrePts, gpuNextPts,gpuStatus,gpuEigenvec;


typedef struct
{
    int i0, i1;
    float correlation;
}ppair, p_ppair;
//Tracking
__device__ int d_newcount[1],d_framewidth[1],d_frameheight[1];
unsigned char * d_rgbframedata,*d_mask,*d_roimask,*d_segmask,*d_segNeg;
float* d_persMap,* d_corners;
float *d_curvec;

//Grouping
__device__ int d_pairN[1];
unsigned char *d_neighbor,* d_neighborD;
unsigned int *d_netOrder;
float* d_distmat, *d_group, *d_netUpdate, *d_crossDist;
ofv* d_ofvec;
ppair* d_pairvec;
int *d_idxmap,* d_overlap,* d_prelabel,* d_label;

unsigned int * h_netOrder;
float *h_netUpdate;
int *tmpn,*idxmap;

cublasHandle_t handle;

__global__ void applyPersToMask(unsigned char* d_mask,float* d_curvec,float* d_persMap)
{
    int pidx=blockIdx.x;
    float px=d_curvec[pidx*2],py=d_curvec[pidx*2+1];
    int blocksize = blockDim.x;
    int w=d_framewidth[0],h=d_frameheight[0];
    int localx = threadIdx.x,localy=threadIdx.y;
    int pxint = px+0.5,pyint = py+0.5;
    float persval =d_persMap[pyint*w+pxint];
    float range=persval/6;
    int offset=range+0.5;
    int yoffset = localy-blocksize/2;
    int xoffset = localx-blocksize/2;
    if(abs(yoffset)<range&&abs(xoffset)<range)
    {
        int globalx=xoffset+pxint,globaly=yoffset+pyint;
        d_mask[globaly*d_framewidth[0]+globalx]=0;
    }
}
__global__ void applySegMask(unsigned char* d_mask,unsigned char* d_segmask,unsigned char* d_segNeg)
{
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    int w=d_framewidth[0],h=d_frameheight[0];
    int totallen =w*h;
    int y=offset/w;
    int x=offset%w;
    if(offset<totallen&&!d_segNeg[offset]&&!d_segmask[offset])
    {
        d_mask[offset]=0;
    }
}
__global__ void renderFrame(unsigned char* d_mask,unsigned char* d_frameptr,float* d_persMap)
{
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    float val=d_persMap[offset]*1000000000;
    int totallen =d_frameheight[0]*d_framewidth[0];

    if(offset<totallen&&d_mask[offset])
    {
        d_frameptr[offset*3]=d_frameptr[offset*3]/2+50;
        d_frameptr[offset*3+1]=d_frameptr[offset*3+1]/2;
        d_frameptr[offset*3+2]=d_frameptr[offset*3+2]/2;
    }
    //d_frameptr[offset*3]=d_frameptr[offset*3]+d_persMap[offset]*1000000000;
}

__global__ void searchNeighbor(unsigned char * d_neighbor, ofv* d_ofvec , ppair* d_pairvec,unsigned int* d_netOrder,float* d_persMap, int nFeatures)
{
    int r = blockIdx.x, c = threadIdx.x;
    if (r < c)
    {
        float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
        int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
        float dist = sqrt(dx*dx + dy*dy);
        int  ymid = (d_ofvec[r].y1 + d_ofvec[c].y1) / 2.0+0.5,xmid = (d_ofvec[r].x1 + d_ofvec[c].x1) / 2+0.5;
        float persval=0;
        persval =d_persMap[ymid*d_framewidth[0]+xmid];
        float hrange=persval+10,wrange=persval+10;
        if(hrange<2)hrange=2;
        if(wrange<2)wrange=2;
        if (dx < wrange && dy < hrange)
        {


            float vx0 = d_ofvec[r].x1 - d_ofvec[r].x0, vx1 = d_ofvec[c].x1 - d_ofvec[c].x0,
                vy0 = d_ofvec[r].y1 - d_ofvec[r].y0, vy1 = d_ofvec[c].y1 - d_ofvec[c].y0;

            float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
            float velocor  = abs(norm0-norm1)/(norm0+norm1);
            float cosine = (vx0*vx1 + vy0*vy1) / (norm0+0.001) / (norm1+0.001);
            int ind = (hrange>dist)-(dist>hrange);
            float cor = (cosine-velocor*2-0.7);// (dist*dist*dist+0.0001);
            if(norm0>0.1&&norm1>0.1)
            {


                //float e=expf(2*cor);
                //cor=(e-1)/(e+1);
                //cor = cor/(1+abs(cor));
                //printf("%f",cor);

            d_neighbor[yidx*nFeatures + xidx] = 1;
            d_neighbor[xidx*nFeatures + yidx] = 1;
            //ppair tmppair;
            //tmppair.i0 = yidx, tmppair.i1 = xidx, tmppair.correlation = cor;
            int arrpos = atomicAdd(d_pairN, 1);
            d_pairvec[arrpos].i0 = yidx, d_pairvec[arrpos].i1 = xidx,d_pairvec[arrpos].correlation=cor;
            //memcpy(d_pairvec + arrpos, &tmppair,sizeof(ppair));
            atomicAdd(d_netOrder + yidx,1);
            atomicAdd(d_netOrder + xidx, 1);
            }
        }
    }
}
__global__ void calUpdate(float* d_group, ppair* d_pairvec, float* d_netUpdate)
{
    int nPair = gridDim.x,nFeatures = blockDim.x;
    int ipair = blockIdx.x, idim = threadIdx.x;
    int i0 = d_pairvec[ipair].i0, i1 = d_pairvec[ipair].i1;
    float cor = d_pairvec[ipair].correlation;
    //printf("%f\n", cor);
    float update0 = d_group[i1*nFeatures + idim] * cor;
    float update1 = d_group[i0*nFeatures + idim] * cor;
    //update0=update0/(1+abs(update0));
    //update1=update1/(1+abs(update1));
    atomicAdd(d_netUpdate + i0*nFeatures + idim, update0);
    atomicAdd(d_netUpdate + i1*nFeatures + idim, update1);
}
__global__ void updateNet(float* d_group, float*  d_netUpdate,unsigned int* d_netOrder)
{
    int idx = blockIdx.x, nFeatures = blockDim.x;
    int dim = threadIdx.x;
    int order = d_netOrder[idx];
    if (order > 0)
    {
        float newval = d_netUpdate[idx*nFeatures+dim]/order;
        d_netUpdate[idx*nFeatures+dim]=newval;
        newval=newval/(1+abs(newval));
        //newval = newval>0;
        float oldval = d_group[idx*nFeatures + dim];
        d_group[idx*nFeatures + dim] = oldval*(1-updateRate) + newval*updateRate;
    }
}

KLTsparse_CUDA::KLTsparse_CUDA()
{
    frame_width=0, frame_height=0;
    frameidx=0;
    nFeatures=0,nSearch=0; 
    /**cuda **/
    persDone=false;
}
KLTsparse_CUDA::~KLTsparse_CUDA()
{
    tracker->releaseMemory();
    detector->releaseMemory();
    gpuGray.release();
    gpuPreGray.release();
    rgbMat.release();
    gpuCorners.release(); 
    gpuPrePts.release();
    gpuNextPts.release();
    gpuStatus.release();
    gpuEigenvec.release();
    cudaFree(d_curvec);
}
int KLTsparse_CUDA::init(int w, int h,unsigned char* framedata,int nPoints)
{
    int nDevices;
    int maxthread=0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        /*
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
        std::cout<<"maxThreadsPerBlock:"<<prop.maxThreadsPerBlock<<std::endl;
        */

        //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,MyKernel, 0, arrayCount);
        if(maxthread==0)maxthread=prop.maxThreadsPerBlock;
        //std::cout << prop.major << "," << prop.minor << std::endl;
    }

    nFeatures = maxthread;//(maxthread>1024)?1024:maxthread;
    nFeatures = (maxthread>nPoints)?nPoints:maxthread;
    nSearch=nFeatures;
    trackBuff = std::vector<FeatBuff>(nFeatures);
    for (int i=0;i<nFeatures;i++)
    {
        trackBuff[i].init(1,100);
    }
    bbTrkBUff.init(1,125);
    frame_width = w,frame_height = h;
    gpu_zalloc(d_persMap, frame_width*frame_height, sizeof(float));
    gpuPersMap= gpu::GpuMat(frame_height, frame_width, CV_32F ,d_persMap);

    h_persMap =  (float*)zalloc(frame_width*frame_height, sizeof(float));
    cudaMemcpyToSymbol(d_framewidth,&frame_width,sizeof(int));
    cudaMemcpyToSymbol(d_frameheight,&frame_height,sizeof(int));
    frameidx=0;
    //detector=new  gpu::GoodFeaturesToTrackDetector_GPU(nSearch,1e-30,0,3);
    detector =new TargetFinder(nSearch,1e-30,0,3);
    tracker =new  gpu::PyrLKOpticalFlow();
    tracker->winSize=Size(9,9);
    tracker->maxLevel=3;
    tracker->iters=10;
    gpuGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpuPreGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpu_zalloc(d_rgbframedata,frame_height*frame_width*3,sizeof(unsigned char));
    rgbMat = gpu::GpuMat(frame_height, frame_width, CV_8UC3 ,d_rgbframedata);
    gpu_zalloc(d_corners,2*nSearch,sizeof(float));
    gpuCorners=gpu::GpuMat(1, nSearch, CV_32FC2,d_corners);
    gpu_zalloc(d_mask,frame_height*frame_width,sizeof(unsigned char));
    maskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,d_mask);
    gpu_zalloc(d_roimask,frame_height*frame_width,sizeof(unsigned char));
    roiMaskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,d_roimask);
    gpu_zalloc(d_segmask,frame_height*frame_width,sizeof(unsigned char));
    gpuSegMat =gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,d_segmask);
    gpu_zalloc(d_segNeg,frame_height*frame_width,sizeof(unsigned char));


    h_roimask =  (unsigned char *)zalloc( frame_height*frame_width,sizeof(unsigned char));
    h_curvec = (float*)zalloc(nFeatures*2,sizeof(float));
    gpu_zalloc(d_curvec, nFeatures * 2, sizeof(float));

    gpu_zalloc(d_neighbor,nFeatures*nFeatures,1);
    h_neighbor = (unsigned char*)zalloc(nFeatures*nFeatures,1);
    gpu_zalloc(d_ofvec, nFeatures, sizeof(ofv));
    ofvBuff.init(1, nFeatures);
    gpu_zalloc(d_group, nFeatures*nFeatures,sizeof(float));
    h_group = (float *)malloc(nFeatures*nFeatures*sizeof(float));

    gpu_zalloc(d_pairvec, nFeatures*nFeatures,sizeof(ppair));
    gpu_zalloc(d_netOrder, nFeatures,sizeof(unsigned int));
    h_netOrder= (unsigned int *)malloc(nFeatures*sizeof(unsigned int));
    gpu_zalloc(d_netUpdate, nFeatures*nFeatures,sizeof(float));
    h_netUpdate = (float*)zalloc(nFeatures*nFeatures,sizeof(float));
    h_pairN = 0;
    cudaMemcpyToSymbol(d_pairN, &h_pairN, sizeof(int));

    h_KnnIdx = (int*)zalloc(KnnK,sizeof(int));
    tmpn = (int*)zalloc(nFeatures,sizeof(int));
    idxmap= (int*)zalloc(nFeatures,sizeof(int));

    gpu_zalloc(d_idxmap,nFeatures,sizeof(int));
    h_prelabel = (int*)zalloc(nFeatures,sizeof(int));
    h_label = (int*)zalloc(nFeatures,sizeof(int));
    label_final =(int*)zalloc(nFeatures,sizeof(int));
    h_gcount = (int*)zalloc(nFeatures,sizeof(int));
    h_clrvec = (unsigned char*)zalloc(nFeatures*3,1);
    items.reserve(nFeatures);

    calPolyGon=false;
    setPts = std::vector< std::vector<cvxPnt> >(nFeatures);
    cvxPts =std::vector< FeatBuff >(nFeatures);
    curK=0,groupN=0,maxgroupN=0;
    h_overlap = (int*)zalloc(nFeatures*nFeatures,sizeof(int));
    h_com = (float*)zalloc(nFeatures*2,sizeof(float));
    updateFlag=false;
    curTrkingIdx=0;
    for(int i=0;i<nFeatures;i++)
    {
        cvxPts[i].init(1,nFeatures);
    }

    selfinit(framedata);
    std::cout<< "inited" << std::endl;
    return 1;
}
int KLTsparse_CUDA::selfinit(unsigned char* framedata)
{
    Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
    rgbMat.upload(curframe);
    gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);
    gpuGray.copyTo(gpuPreGray);
    (*detector)(gpuGray, gpuCorners);
    gpuCorners.download(corners);
    gpuCorners.copyTo(gpuPrePts);
    for (int k = 0; k < nFeatures; k++)
    {
        Vec2f p = corners.at<Vec2f>(k);
        pttmp.x = p[0];
        pttmp.y = p[1];
        pttmp.t = frameidx;
        trackBuff[k].updateAFrame(&pttmp);
        memset(h_group + k*nFeatures, 0, nFeatures*sizeof(float));
        h_group[k*nFeatures + k] = 1;
        h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
        h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
    }
    cudaMemset(d_mask,255,frame_width*frame_height*sizeof(unsigned char));
    cudaMemset(d_roimask,255,frame_width*frame_height*sizeof(unsigned char));
    return true;
}
void KLTsparse_CUDA::setUpPersMap(float* srcMap)
{
    memcpy(h_persMap,srcMap,frame_width*frame_height*sizeof(float));
    cudaMemcpy(d_persMap,srcMap,frame_width*frame_height*sizeof(float),cudaMemcpyHostToDevice);
    detector->setPersMat(gpuPersMap,frame_width,frame_height);
}
bool KLTsparse_CUDA::checkTrackMoving(FeatBuff &strk)
{
    bool isTrkValid = true;
    if(strk.len>1)
    {
        PntT xb=strk.cur_frame_ptr->x,yb=strk.cur_frame_ptr->y;
        float persval = h_persMap[yb*frame_width+xb];
        PntT prex=strk.getPtr(strk.len-2)->x, prey=strk.getPtr(strk.len-2)->y;
        double trkdist=abs(prex-xb)+abs(prey-yb);
        if(trkdist>persval)return false;
        int Movelen=150/sqrt(persval),startidx=max(strk.len-Movelen,0);
        if(strk.len>Movelen)
        {
            FeatPts* aptr = strk.getPtr(startidx);
            PntT xa=aptr->x,ya=aptr->y;
            double displc=sqrt((xb-xa)*(xb-xa) + (yb-ya)*(yb-ya));
            if((strk.len -startidx)*MoveFactor>displc)
            {
                isTrkValid = false;
            }
        }


    }
    return isTrkValid;
}
void KLTsparse_CUDA::updateSegCPU(unsigned char* ptr)
{
    //Mat kernel=Mat::ones(5,5,CV_8UC1);
    cudaMemcpy(d_segmask,ptr,frame_height*frame_width,cudaMemcpyHostToDevice);
    //dilate(gpuSegMat, gpuDiaMat, kernel, Point(-3, -3));
    //cudaMemcpy(d_segmask,gpuDiaMat.data,frame_height*frame_width,cudaMemcpyHostToDevice);

}
void KLTsparse_CUDA::updateROICPU(float* aryPtr,int length)
{
    cudaMemset(d_roimask,0,frame_height*frame_width*sizeof(unsigned char));
    memset(h_roimask,0,frame_height*frame_width*sizeof(unsigned char));
    std::vector<Point2f> roivec;
    int counter=0;
    for(int i=0;i<length;i++)
    {
        Point2f p(aryPtr[i*2],aryPtr[i*2+1]);
        roivec.push_back(p);
    }
    for(int i=0;i<frame_height;i++)
    {
        for(int j=0;j<frame_width;j++)
        {
            if(pointPolygonTest(roivec,Point2f(j,i),true)>0)
            {
                h_roimask[i*frame_width+j]=255;
                counter++;

            }
        }
    }
    std::cout<<counter<<std::endl;
    cudaMemcpy(d_roimask,h_roimask,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,d_roimask,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyDeviceToDevice);
}
void KLTsparse_CUDA::updateSegNeg(float* aryPtr,int length)
{
    unsigned char * h_segNeg = (unsigned char *)zalloc( frame_height*frame_width,sizeof(unsigned char));
    cudaMemset(d_segNeg,0,frame_height*frame_width*sizeof(unsigned char));
    memset(h_segNeg,0,frame_height*frame_width*sizeof(unsigned char));
    std::vector<Point2f> roivec;
    int counter=0;
    for(int i=0;i<length;i++)
    {
        Point2f p(aryPtr[i*2],aryPtr[i*2+1]);
        roivec.push_back(p);
    }
    for(int i=0;i<frame_height;i++)
    {
        for(int j=0;j<frame_width;j++)
        {
            if(pointPolygonTest(roivec,Point2f(j,i),true)>0)
            {
                h_segNeg[i*frame_width+j]=255;
                counter++;

            }
        }
    }
    std::cout<<counter<<std::endl;
    cudaMemcpy(d_segNeg,h_segNeg,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyHostToDevice);
}
void KLTsparse_CUDA::findPoints()
{
    std::cout<<"applySegMask"<<std::endl;
    if(applyseg)
    {
        int nblocks = (frame_height*frame_width)/nFeatures;
        applySegMask<<<nblocks,nFeatures>>>(d_mask,d_segmask,d_segNeg);
    }
    std::cout<<"detector"<<std::endl;
    (*detector)(gpuGray, gpuCorners,maskMat);
    gpuCorners.download(corners);
}
void KLTsparse_CUDA::filterTrack()
{
    int addidx=0,lostcount=0;
    std::cout<<"for loop"<<std::endl;
    for (int k = 0; k < nFeatures; k++)
    {
        std::cout<<"Track:"<<k<<std::endl;
        int statusflag = status.at<int>(k);
        Vec2f trkp = nextPts.at<Vec2f>(k);
        bool lost=false;
        bool ismoving=false;
        if ( statusflag)
        {
            pttmp.x = trkp[0];
            pttmp.y = trkp[1];
            pttmp.t = frameidx;
            trackBuff[k].updateAFrame(&pttmp);
            std::cout<<"checkTrackMoving"<<std::endl;
            ismoving = checkTrackMoving(trackBuff[k]);
            std::cout<<"Done checkTrackMoving"<<std::endl;
            if (!ismoving)lost=true;
        }
        else
        {
            lost=true;
        }
        if(lost)
        {
            trackBuff[k].clear();
            label_final[k]=0;
            memset(h_group + k*nFeatures, 0, nFeatures*sizeof(float));
            h_label[k]=0;
            if(lostcount<corners.size[1])
            {
                Vec2f cnrp = corners.at<Vec2f>(lostcount++);
                pttmp.x = cnrp[0];
                pttmp.y = cnrp[1];
                pttmp.t = frameidx;
                trackBuff[k].updateAFrame(&pttmp);
                nextPts.at<Vec2f>(k)=cnrp;

                h_group[k*nFeatures + k] = 1;
            }
        }
        int x =trackBuff[k].cur_frame_ptr->x,
                y=trackBuff[k].cur_frame_ptr->y;
        h_curvec[addidx * 2] = trackBuff[k].cur_frame_ptr->x;
        h_curvec[addidx * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
        addidx++;
        if (ismoving&&trackBuff[k].len > minTrkLen)
        {

            ofvtmp.x0 = trackBuff[k].getPtr(trackBuff[k].len - minTrkLen-1)->x;
            ofvtmp.y0 = trackBuff[k].getPtr(trackBuff[k].len - minTrkLen-1)->y;
            ofvtmp.x1 = trackBuff[k].cur_frame_ptr->x;
            ofvtmp.y1 = trackBuff[k].cur_frame_ptr->y;
            ofvtmp.len = trackBuff[k].len;
            ofvtmp.idx = k;
            ofvBuff.updateAFrame(&ofvtmp);
            items.push_back(k);

        }
    }
    std::cout<<"applyPersToMask"<<std::endl;
    cudaMemcpy(d_curvec, h_curvec, nFeatures*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,d_roimask,frame_height*frame_width*sizeof(unsigned char),cudaMemcpyDeviceToDevice);
    dim3 block(32, 32,1);
    applyPersToMask<<<addidx,block>>>(d_mask,d_curvec, d_persMap);
    gpuPrePts.upload(nextPts);
}
void KLTsparse_CUDA::PointTracking()
{
    std::cout<<"tracker"<<std::endl;
    tracker->sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus);
    gpuStatus.download(status);
    gpuNextPts.download(nextPts);
}
int KLTsparse_CUDA::updateAframe(unsigned char* framedata, int fidx)
{
    frameidx=fidx;
    std::cout<<"frameidx:"<<frameidx<<std::endl;
    gpuGray.copyTo(gpuPreGray);
    std::cout<<"gpuPreGray"<<std::endl;
    Mat curframe(frame_height,frame_width,CV_8UC3,framedata);

    rgbMat.upload(curframe);

    gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);
    PointTracking();
    findPoints();
    std::cout<<"renderFrame"<<std::endl;
    if(render)
    {
        int nblocks = (frame_height*frame_width)/nFeatures;
        renderFrame<<<nblocks,nFeatures>>>(d_mask,rgbMat.data,(float * )detector->eig_.data);
        cudaMemcpy(framedata,rgbMat.data,frame_height*frame_width*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    }
    ofvBuff.clear();
    items.clear();
    filterTrack();

    /** Grouping  **/
    if(h_gcount[curTrkingIdx]<1)updateFlag=true;
    if(groupOnFlag&&ofvBuff.len>0)
    {

        h_pairN = 0;

        cudaMemset(d_ofvec, 0, nFeatures* sizeof(ofv));

        cudaMemcpy(d_ofvec, ofvBuff.data, ofvBuff.len*sizeof(ofv), cudaMemcpyHostToDevice);

        //cudaMemcpy(d_neighbor,h_neighbor , nFeatures*nFeatures*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_neighbor, 0, nFeatures*nFeatures);

        cudaMemset(d_netOrder, 0, nFeatures*sizeof(unsigned int));

        cudaMemcpyToSymbol(d_pairN, &h_pairN, sizeof(int));

        searchNeighbor <<<ofvBuff.len, ofvBuff.len >>>(d_neighbor, d_ofvec, d_pairvec,d_netOrder,d_persMap,nFeatures);

        cudaMemcpy(h_netOrder,d_netOrder,nFeatures*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_neighbor, d_neighbor, nFeatures*nFeatures, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_pairN, d_pairN, sizeof(int));
        cudaMemcpy(d_group, h_group, nFeatures*nFeatures*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_netUpdate, 0, nFeatures*nFeatures*sizeof(float));

        calUpdate <<<h_pairN, nFeatures>>>(d_group, d_pairvec, d_netUpdate);
        cudaDeviceSynchronize();
        updateNet <<<nFeatures,nFeatures>>>(d_group, d_netUpdate, d_netOrder);
        cudaMemcpy(h_netUpdate, d_netUpdate, nFeatures*nFeatures*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_group, d_group, nFeatures*nFeatures*sizeof(float), cudaMemcpyDeviceToHost);

        float maxval=0.1,avgval=0;
        int count=0;
        for (int i = 0; i < nFeatures; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                //std::cout <<h_group[i*nFeatures + j]<<",";
                if(j!=i)
                {
                    //if(h_netUpdate[i*nFeatures + j]>0)std::cout<<h_netUpdate[i*nFeatures + j]<<",";
                    //if(h_group[i*nFeatures + j]>0.0){
                    if(h_group[i*nFeatures + j]>0)
                    {
                    avgval+=h_group[i*nFeatures + j];
                    count++;
                    }
                int ind = (maxval>h_group[i*nFeatures + j]);
                maxval = ind*maxval + (1 - ind)*h_group[i*nFeatures + j];//}

                }
            }

        }
        avgval/=count;
        std::cout <<avgval<< std::endl;

        for (int i = 0; i < nFeatures; i++)
        {
            for(int j = i; j < nFeatures; j++)
            {
                if(h_neighbor[i*nFeatures + j]&&h_group[i*nFeatures + j]>CoThresh*avgval&&h_group[j*nFeatures + i]>CoThresh*avgval)
                {
                    h_neighbor[i*nFeatures + j]=2;
                    h_neighbor[j*nFeatures + i]=2;
                }
            }
        }

        /** Single Frame Grouping(BFS)
         * then map to previous Group Idx Acoording to OverLap  **/
        memcpy(h_prelabel ,h_label,nFeatures*sizeof(int));
        pregroupN = groupN;
        bfsearch();
        memset(h_overlap,0,nFeatures*nFeatures*sizeof(int));
        for(int i=0;i<nFeatures;i++)
        {
            int prelabel = h_prelabel[i],label = h_label[i];
            if(prelabel&&label)
                h_overlap[prelabel*nFeatures+label]++;
        }
        reGroup();
        for(int i = 0;i<nFeatures;i++)
        {
            if(h_label[i])
            {
                h_label[i]=idxmap[h_label[i]];
            }
        }

        /** Group Smoothing **/
        memcpy(label_final,h_label,nFeatures*sizeof(int));


        /**  Calculate Group Properties **/
        int maxidx=0;
        memset(h_gcount,0,nFeatures*sizeof(int));
        memset(h_com,0,nFeatures*2*sizeof(float));
        if(calPolyGon)
        {
            for(int i=0;i<=maxgroupN;i++)
            {
                setPts[i].clear();
                cvxPts[i].clear();
            }
        }
        for(int i=0;i<nFeatures;i++)
        {
            int gidx = label_final[i];
            if(gidx)
            {
                h_gcount[gidx]++;
                if(calPolyGon)
                {
                    cvxPnttmp.x=h_curvec[i*2];
                    cvxPnttmp.y=h_curvec[i*2+1];
                    setPts[gidx].push_back(cvxPnttmp);
                }
                h_com[gidx*2]+=trackBuff[i].cur_frame_ptr->x;
                h_com[gidx*2+1]+=trackBuff[i].cur_frame_ptr->y;
                if(gidx>maxidx)maxidx=gidx;
            }
        }

        if(maxidx>maxgroupN)maxgroupN=maxidx+1;
        for(int i=1;i<=maxgroupN;i++)
        {
            if(h_gcount[i]>0)
            {
                HSVtoRGB(h_clrvec+i*3,h_clrvec+i*3+1,h_clrvec+i*3+2,i/(maxgroupN+0.01)*360,1,1);
                h_com[i*2]/=float(h_gcount[i]);
                h_com[i*2+1]/=float(h_gcount[i]);
                if(calPolyGon)convex_hull(setPts[i],cvxPts[i]);
            }
        }
    }
    return 1;
}
void KLTsparse_CUDA::reGroup()
{
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));

    int vaccount=0;
    for(int i=1;i<=pregroupN;i++)
    {
        int maxcount=0,maxidx=0;
        for(int j=1;j<=groupN;j++)
        {
            if( h_overlap[i*nFeatures+j]>maxcount)
            {
                maxidx = j;
                maxcount = h_overlap[i*nFeatures+j];
            }
        }
        if(maxidx)
        {
            if(idxmap[maxidx])
            {
                if(h_overlap[i*nFeatures+maxidx]>h_overlap[idxmap[maxidx]*nFeatures+maxidx])
                {
                    tmpn[vaccount++]=idxmap[maxidx];
                    idxmap[maxidx]=i;
                }
                else
                {
                    tmpn[vaccount++]=i;
                }
            }
            else
                idxmap[maxidx]=i;
        }
        else
        {
            tmpn[vaccount++]=i;
        }
    }
    int vci=0;
    for(int i=1;i<=groupN;i++)
    {
        if(!idxmap[i])
        {
            if(vci<vaccount)
                idxmap[i]=tmpn[vci++];
            else
                idxmap[i]=(++pregroupN);
        }
    }
}
void KLTsparse_CUDA::bfsearch()
{
    int pos=0;
    bool isempty=false;
    int gcount=0;
    curK=1;
    groupN=0;
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));
    memset(h_label,0,nFeatures*sizeof(int));
    memset(h_gcount,0,nFeatures*sizeof(int));
    int idx = items[pos];
    h_label[idx]=curK;
    for(int i=0;i<nFeatures;i++)
    {
        tmpn[i]=(h_neighbor[idx*nFeatures+i]>1);
    }
    items[pos]=0;
    gcount++;
    while (!isempty) {
        isempty=true;
        int ii=0;
        for(pos=0;pos<items.size();pos++)
        {
            idx=items[pos];
            if(idx)
            {
                if(ii==0)ii=pos;
                isempty=false;
                if(tmpn[idx])
                {
                    int nc=0,nnc=0;
                    //nc neighbor count
                    //nnc sec tier neighbor count
                    for(int i=0;i<nFeatures;i++)
                    {
                        if(h_neighbor[idx*nFeatures+i]>1)
                        {
                            nc++;
                            //if(tmpn[i])nnc++;
                            nnc+=(tmpn[i]>0);
                        }
                    }
                    //if(nnc>0)
                    {
                        gcount++;
                        h_label[idx]=curK;
                        for(int i=0;i<nFeatures;i++)
                        {
                            tmpn[i]+=(h_neighbor[idx*nFeatures+i]>1);
                        }
                        items[pos]=0;
                        if(ii==pos)ii=0;
                    }
                }
            }
        }
        if(gcount>0)
        {
            h_gcount[curK]+=gcount;
            gcount=0;
        }
        else if(!isempty)
        {
            if(h_gcount[curK]>minGSize)
            {
                groupN++;
                idxmap[curK]=groupN;
            }
            curK++;
            gcount=0;
            memset(tmpn,0,nFeatures*sizeof(int));
            pos=ii;
            idx=items[pos];
            gcount++;
            h_label[idx]=curK;
            for(int i=0;i<nFeatures;i++)
            {
                tmpn[i]+=(h_neighbor[idx*nFeatures+i]>1);
            }
            items[pos]=0;
        }
    }
    for(int i=0;i<nFeatures;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }

}
