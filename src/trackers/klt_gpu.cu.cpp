#include "trackers/klt_gpu.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
using namespace cv;
//using namespace std;
#define PI 3.14159265

#define minDist 2
#define minGSize 2
#define TIMESPAN 15
#define COSTHRESH 0.4
#define VeloThresh 0.1
#define KnnK 40
#define MoveFactor 0.000001
#define coNBThresh 0
#define minTrkLen 2


#define gpu_zalloc(ptr, num, size) cudaMalloc(&ptr,size*num);cudaMemset(ptr,0,size*num);

Mat corners,prePts,nextPts,status,eigenvec;
cv::gpu::GoodFeaturesToTrackDetector_GPU* detector;
cv::gpu::PyrLKOpticalFlow* tracker;
cv::gpu::GpuMat gpuGray, gpuPreGray,rgbMat, gpuCorners, gpuPrePts, gpuNextPts,gpuStatus,gpuEigenvec;


typedef struct
{
    int i0, i1;
    float correlation;
}ppair, p_ppair;

int *tmpn,*idxmap;

__device__ float pershk[1],pershb[1],perswk[1],perswb[1];
__device__ int d_newcount[1],d_framewidth[1],d_frameheight[1];
float * persMap;
unsigned char* d_isnewmat, *d_neighbor,* d_neighborD;
float* d_cosine,*d_velo;
ofv* d_ofvec;
float *d_curvec,*d_distmat;
float* d_persMap;
int *d_newidx,*d_idxmap;
cublasHandle_t handle;

__global__ void crossDistBox(unsigned char* dst,float* vertical,float* horizon,int h,int w)
{
    int x = threadIdx.x,y=blockIdx.x;
    float xv = vertical[y * 2], yv = vertical[y*2+1],xh=horizon[x*2],yh=horizon[x*2+1];
    float dx = abs(xv - xh), dy = abs(yv - yh),ymid=(yv+yh)/2.0;
    float hrange=(ymid*pershk[0]/2+pershb[0])/10,wrange=ymid*perswk[0]/2+perswb[0]/10;
    if((dx<wrange&&dy<hrange)||(abs(dx)<1&&abs(dx)<1))dst[y]=1;
}
__global__ void crossDistMap(unsigned char* dst,float* vertical,float* horizon,float * d_persMap)
{
    int x = threadIdx.x,y=blockIdx.x;
    float xv = vertical[y * 2], yv = vertical[y*2+1],xh=horizon[x*2],yh=horizon[x*2+1];
    float dx = abs(xv - xh), dy = abs(yv - yh);
    float   persval=0;
    int ymid=(yv+yh)/2.0+0.5,xmid=(xv+xh)/2.0+0.5;
    if(ymid<d_frameheight[0]&&xmid<d_framewidth[0])
    {
//        ymid=(ymid<d_frameheight[0])*ymid;//+(ymid>=d_frameheight[0])*(d_frameheight[0]-1);
//        xmid=(xmid<d_framewidth[0])*xmid;//+(xmid>=d_framewidth[0])*(d_framewidth[0]-1);
//        ymid=(ymid>0)*ymid;
//        xmid=(xmid>0)*xmid;


        persval =d_persMap[ymid*d_framewidth[0]+xmid];
    }
    float hrange=persval/10,wrange=persval/10;
    if((dx<wrange&&dy<hrange)||(abs(dx)<1&&abs(dy)<1))dst[y]=1;
}

__global__ void findZero(unsigned char* d_isnewmat,int* d_newidx,int nSearch)
{
    int stripe=blockDim.x;
    int idx=blockIdx.x*stripe+threadIdx.x;
    if(idx<nSearch&&!d_isnewmat[idx])
    {
        int arrpos = atomicAdd(d_newcount, 1);
        d_newidx[arrpos]=idx;
    }
}

__global__ void searchNeighborBox(unsigned char* d_neighbor,float* d_cosine,float* d_velo, ofv* d_ofvec,float* d_distmat ,int offsetidx, int nFeatures)
{
    int r = blockIdx.x, c = threadIdx.x;
    if (r < c)
    {
        unsigned char* curptr=d_neighbor;
        float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
        int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
        float  ymid = (d_ofvec[r].y1 + d_ofvec[c].y1) / 2;//xmid = (d_ofvec[r].x1 + d_ofvec[c].x1) / 2,
        float hrange=(ymid*pershk[0]+pershb[0]),wrange=(ymid*perswk[0]+perswb[0]);

        if(hrange<2)hrange=2;
        if(wrange<2)wrange=2;
        if (dx < wrange && dy < hrange)
        {
            curptr[yidx*nFeatures+xidx]=1;
            curptr[xidx*nFeatures+yidx]=1;
            float vx0 = d_ofvec[r].x1 - d_ofvec[r].x0, vx1 = d_ofvec[c].x1 - d_ofvec[c].x0,
                vy0 = d_ofvec[r].y1 - d_ofvec[r].y0, vy1 = d_ofvec[c].y1 - d_ofvec[c].y0;
            float dist = dx*dx+dy*dy;
            float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
            float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;
            d_velo[(yidx*nFeatures+xidx)*2]=vx0;
            d_velo[(yidx*nFeatures+xidx)*2+1]=vy0;
            d_velo[(xidx*nFeatures+yidx)*2]=vx1;
            d_velo[(xidx*nFeatures+yidx)*2+1]=vy1;
            d_cosine[yidx*nFeatures+xidx]=cosine;
            d_cosine[xidx*nFeatures+yidx]=cosine;
            d_distmat[yidx*nFeatures+xidx]=dist;
            d_distmat[xidx*nFeatures+yidx]=dist;
        }
    }
}

__global__ void searchNeighborMap(unsigned char* d_neighbor,float* d_cosine,float* d_velo, ofv* d_ofvec,float* d_distmat ,float * d_persMap, int nFeatures)
{
    int r = blockIdx.x, c = threadIdx.x;
    if (r < c)
    {
        unsigned char* curptr=d_neighbor;
        float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
        int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
        int  ymid = (d_ofvec[r].y1 + d_ofvec[c].y1) / 2.0+0.5,xmid = (d_ofvec[r].x1 + d_ofvec[c].x1) / 2+0.5;
        float persval=0;
        if(ymid<d_frameheight[0]&&xmid<d_framewidth[0])
            persval =d_persMap[ymid*d_framewidth[0]+xmid];
        float hrange=persval/10,wrange=persval/10;

        if(hrange<2)hrange=2;
        if(wrange<2)wrange=2;
        if (dx < wrange && dy < hrange)
        {
            curptr[yidx*nFeatures+xidx]=1;
            curptr[xidx*nFeatures+yidx]=1;
            float vx0 = d_ofvec[r].x1 - d_ofvec[r].x0, vx1 = d_ofvec[c].x1 - d_ofvec[c].x0,
                vy0 = d_ofvec[r].y1 - d_ofvec[r].y0, vy1 = d_ofvec[c].y1 - d_ofvec[c].y0;
            float dist = dx*dx+dy*dy;
            float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
            float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;
            d_velo[(yidx*nFeatures+xidx)*2]=vx0;
            d_velo[(yidx*nFeatures+xidx)*2+1]=vy0;
            d_velo[(xidx*nFeatures+yidx)*2]=vx1;
            d_velo[(xidx*nFeatures+yidx)*2+1]=vy1;
            d_cosine[yidx*nFeatures+xidx]=cosine;
            d_cosine[xidx*nFeatures+yidx]=cosine;
            d_distmat[yidx*nFeatures+xidx]=dist;
            d_distmat[xidx*nFeatures+yidx]=dist;
        }
    }
}

__global__ void neighborD(unsigned char* d_neighbor,float* d_cosine,float* d_velo,unsigned char* d_neighborD,int offsetidx)
{
    int yidx = blockIdx.x, xidx = threadIdx.x,nFeatures = blockDim.x;
    unsigned char val = 1;
    float cosine = 0,norm0 =0,norm1=0,vx0=0,vy0=0,vx1=0,vy1=0;
    for(int i=0;i<TIMESPAN;i++)
    {
        val=val&&d_neighbor[i*nFeatures*nFeatures+yidx*nFeatures+xidx];
        cosine+=d_cosine[i*nFeatures*nFeatures+yidx*nFeatures+xidx];
        vx0+=d_velo[(i*nFeatures*nFeatures+yidx*nFeatures+xidx)*2];
        vy0+=d_velo[(i*nFeatures*nFeatures+yidx*nFeatures+xidx)*2+1];
        vx1+=d_velo[(i*nFeatures*nFeatures+xidx*nFeatures+yidx)*2];
        vy1+=d_velo[(i*nFeatures*nFeatures+xidx*nFeatures+yidx)*2+1];
    }
    cosine/=(TIMESPAN+1);
    vx0/=(TIMESPAN+1);
    vy0/=(TIMESPAN+1);
    vx1/=(TIMESPAN+1);
    vy1/=(TIMESPAN+1);
    norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
    float veloVar= abs(norm0-norm1)/(norm0+norm1);
    if(val&&cosine>COSTHRESH&&veloVar<VeloThresh)
        d_neighborD[yidx*nFeatures+xidx]=1;
}
__device__ void d_HSVtoRGB( unsigned char *r, unsigned char *g, unsigned char *b, float h, float s, float v )
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


KLTsparse_CUDA::KLTsparse_CUDA()
{
    frame_width=0, frame_height=0;
    frameidx=0;
    nFeatures=0,nSearch=0; 
    phk=0.282333,phb=-47.4776,pwk=0.121724,pwb=-9.09214;
    /**cuda **/
    offsetidx=0;
    curTrkingIdx=0;
    updateFlag=false,calPolyGon=false,persDone=false;
    h_newcount=0,curK=0,pregroupN=0,groupN=0,maxgroupN=0;
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
    cudaFree(d_isnewmat);
    cudaFree(d_newidx);
    cudaFree(d_newcount);
    cudaFree(d_neighbor);
    cudaFree(d_neighborD);
    cudaFree(d_cosine);
    cudaFree(d_velo);
    cudaFree(d_ofvec);
    cudaFree(d_curvec);
    cudaFree(d_distmat);
    cudaFree(d_idxmap);
}
int KLTsparse_CUDA::init(int w, int h,unsigned char* framedata)
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
        if(maxthread==0)maxthread=prop.maxThreadsPerBlock;
        //std::cout << prop.major << "," << prop.minor << std::endl;
    }

    nFeatures = (maxthread>1024)?1024:maxthread;
    nSearch=nFeatures*2;
    std::cout<<nFeatures<<std::endl;
    trackBuff = std::vector<FeatBuff>(nFeatures);
    for (int i=0;i<nFeatures;i++)
    {
        trackBuff[i].init(1,100);
    }
    bbTrkBUff.init(1,125);
    frame_width = w;
    frame_height = h;
    frameidx=0;
    detector=new  gpu::GoodFeaturesToTrackDetector_GPU(nSearch,0.0001,0,7);
    tracker =new  gpu::PyrLKOpticalFlow();
    tracker->winSize=Size(7,7);
    tracker->maxLevel=3;
    tracker->iters=10;
    gpuGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpuPreGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    rgbMat = gpu::GpuMat(frame_height, frame_width, CV_8UC3 );
    nextPts=cv::Mat(1,nFeatures,CV_32FC2);
    prePts=cv::Mat(1,nFeatures,CV_32FC2);
    gpuPrePts=gpu::GpuMat(1,nFeatures,CV_32FC2);
    gpuNextPts=gpu::GpuMat(1,nFeatures,CV_32FC2);
    gpu_zalloc(d_isnewmat, nSearch,1);
    h_newidx = (int*)zalloc(nSearch,sizeof(int));
    gpu_zalloc(d_newidx, nSearch,sizeof(int));
    h_newcount=0;
    cudaMemcpyToSymbol(d_newcount,&h_newcount,sizeof(int));
    h_curvec = (float*)zalloc(nFeatures*2,sizeof(float));
    gpu_zalloc(d_curvec, nFeatures * 2, sizeof(float));
    offsetidx=0;
    gpu_zalloc(d_neighbor,nFeatures*nFeatures*TIMESPAN,1);
    gpu_zalloc(d_cosine,nFeatures*nFeatures*TIMESPAN,sizeof(float));
    gpu_zalloc(d_velo,nFeatures*nFeatures*2*TIMESPAN,sizeof(float));
    gpu_zalloc(d_ofvec, nFeatures, sizeof(ofv));
    ofvBuff.init(1, nFeatures);
    gpu_zalloc(d_distmat,nFeatures*nFeatures,sizeof(float));
    h_distmat = (float *)zalloc(nFeatures*nFeatures,sizeof(float));
    h_KnnIdx = (int*)zalloc(KnnK,sizeof(int));
    gpu_zalloc(d_neighborD,nFeatures*nFeatures,1);
    h_curnbor = (unsigned char*)zalloc(nFeatures*nFeatures,1);
    h_neighborD=(unsigned char*)zalloc(nFeatures*nFeatures,1);


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
    h_precom = (float*)zalloc(nFeatures*2,sizeof(float));
    h_bbox = (float*)zalloc(nFeatures*4,sizeof(float));
    h_bbstats = (float*)zalloc(nFeatures*4,sizeof(float));
    updateFlag=false;
    curTrkingIdx=0;
    for(int i=0;i<nFeatures;i++)
    {
        cvxPts[i].init(1,nFeatures);
    }
    selfinit(framedata);
    //std::cout<< "inited" << std::endl;
    return 1;
}
void KLTsparse_CUDA::setUpPers(float l0,float t0,float r0,float b0,float l1,float t1,float r1,float b1)
{
    float y0=(b0+t0)/2.0,h0=(b0-t0),w0=(r0-l0);//,x0=(r0+l0)/2.0
    float y1=(b1+t1)/2.0,h1=(b1-t1),w1=(r1-l1);//,x1=(r1+l1)/2.0
    float hk = (h1-h0)/(y1-y0),hb=h1-y1*hk;
    float wk = (w1-w0)/(y1-y0),wb=w1-y1*wk;
    phk=hk,phb=hb,pwk=wk,pwb=wb;
    cudaMemcpyToSymbol(pershk,&phk,sizeof(float));
    cudaMemcpyToSymbol(pershb,&phb,sizeof(float));
    cudaMemcpyToSymbol(perswk,&pwk,sizeof(float));
    cudaMemcpyToSymbol(perswb,&pwb,sizeof(float));
    //std::cout<<hk<<","<<hb<<"|"<<wk<<","<<wb<<std::endl;
    persDone=true;
}
void KLTsparse_CUDA::setUpPersMap(float* srcMap)
{
    gpu_zalloc(d_persMap, frame_width*frame_height, sizeof(float));
    cudaMemcpy(d_persMap,srcMap,frame_width*frame_height*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_framewidth,&frame_width,sizeof(int));
    cudaMemcpyToSymbol(d_frameheight,&frame_height,sizeof(int));
}
int KLTsparse_CUDA::selfinit(unsigned char* framedata)
{
    Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
    rgbMat.upload(curframe);
    gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);
    gpuGray.copyTo(gpuPreGray);

    (*detector)(gpuGray, gpuCorners);
    std::cout<<"gpuCorners.size():"<<gpuCorners.size()<<std::endl;
    gpuCorners.download(corners);

//    gpuCorners.copyTo(gpuPrePts);/
    std::cout<<"gpuPrePts.size():"<<gpuPrePts.size()<<std::endl;
    for (int k = 0; k < nFeatures; k++)
    {
        Vec2f p = corners.at<Vec2f>(k);
        pttmp.x = p[0];
        pttmp.y = p[1];
        nextPts.at<Vec2f>(k)=p;


        pttmp.t = frameidx;
        trackBuff[k].updateAFrame(&pttmp);
        h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
        h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
    }
    long unsigned int freemem,totmem;
    cuMemGetInfo(&freemem,&totmem);
    std::cout<<freemem/1024/1024<<","<<totmem/1024/1024<<std::endl;
    std::cout<<"nextPts.step():"<<nextPts.step<<"cols:"<<nextPts.cols<<"rows"<<nextPts.rows<<std::endl;
    gpuPrePts.upload(nextPts);
    std::cout<<(intptr_t)gpuPrePts.data<<std::endl;
    //std::cout<<gpuPrePts.data<<std::endl;
    return true;
}
bool KLTsparse_CUDA::checkTrackMoving(FeatBuff &strk)
{
    bool isTrkValid = true;
    int Movelen=7,startidx=max(strk.len-Movelen,0);
    if(strk.len>Movelen)
    {
        FeatPts* aptr = strk.getPtr(startidx);
        PntT xa=aptr->x,ya=aptr->y,xb=strk.cur_frame_ptr->x,yb=strk.cur_frame_ptr->y;
        double displc=sqrt((xb-xa)*(xb-xa) + (yb-ya)*(yb-ya));
        if((strk.len -startidx)*MoveFactor>displc)
        {
            isTrkValid = false;
        }
    }
    return isTrkValid;
}

int KLTsparse_CUDA::updateAframe(unsigned char* framedata, int fidx)
{
    frameidx=fidx;
    std::cout<<frameidx<<std::endl;

    gpuGray.copyTo(gpuPreGray);
    //cudaMemcpy(gpuPreGray.data,gpuGray.data,frame_height*frame_width,cudaMemcpyDeviceToDevice);
    //Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
    //rgbMat.upload(curframe);
    cudaMemcpy(rgbMat.data,framedata,frame_height*frame_width*3,cudaMemcpyHostToDevice);

    gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);
    std::cout<<"start klt matching"<<std::endl;
    tracker->sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus);
    std::cout<<"finished klt matching:"<<gpuNextPts.size()<<std::endl;
    gpuStatus.download(status);
    gpuNextPts.download(nextPts);
    gpuPrePts.download(prePts);
    std::cout<<"start detect"<<std::endl;
    (*detector)(gpuGray, gpuCorners);
    gpuCorners.download(corners);
//    for(int i=0;i<nFeatures;i++)
//    {
//       std::cout<<h_curvec[i*2]<<","<<h_curvec[i*2+1]<<std::endl;
//    }
    std::cout<<"finished detect:"<<corners.size()<<std::endl;
    cudaMemcpy(d_curvec, h_curvec, nFeatures*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_isnewmat, 0, nSearch);
    crossDistBox<<<nSearch, nFeatures>>>(d_isnewmat, (float *)gpuCorners.data, d_curvec, nSearch, nFeatures);
    //crossDistMap<<<nSearch, nFeatures>>>(d_isnewmat, (float *)gpuCorners.data, d_curvec,d_persMap);
    h_newcount=0;
    cudaMemcpyToSymbol(d_newcount,&h_newcount,sizeof(int));
    findZero<<<nSearch/nFeatures+1,nFeatures>>>(d_isnewmat, d_newidx, nSearch);
    cudaMemcpyFromSymbol(&h_newcount, d_newcount, sizeof(int));
    cudaMemcpy(h_newidx, d_newidx, nSearch, cudaMemcpyDeviceToHost);

    int addidx=0;
    ofvBuff.clear();
    items.clear();
    std::cout<<"filter static and Add New "<<std::endl;
    for (int k = 0; k < nFeatures; k++)
    {
        int statusflag = status.at<int>(k);
        Vec2f trkp = nextPts.at<Vec2f>(k);
        bool lost=false;
        if ( statusflag)
        {
            int prex=trackBuff[k].cur_frame_ptr->x, prey=trackBuff[k].cur_frame_ptr->y;
            pttmp.x = trkp[0];
            pttmp.y = trkp[1];
            pttmp.t = frameidx;
            trackBuff[k].updateAFrame(&pttmp);
            double trkdist=abs(prex-pttmp.x)+abs(prey-pttmp.y);
            bool isMoving=checkTrackMoving(trackBuff[k]);
            if (!isMoving||(trackBuff[k].len>1 && trkdist>50))
            {
                lost=true;
            }
        }
        else
        {
            lost=true;
        }
        if(lost)
        {
            //if(trackBuff[k].len>10)saveTrk(trackBuff[k]);
            trackBuff[k].clear();
            label_final[k]=0;
            if(addidx<h_newcount)
            {
                Vec2f cnrp = corners.at<Vec2f>(h_newidx[addidx++]);
                pttmp.x = cnrp[0];
                pttmp.y = cnrp[1];
                pttmp.t = frameidx;
                trackBuff[k].updateAFrame(&pttmp);
                nextPts.at<Vec2f>(k)=cnrp;
            }
        }
        else
        {
            if (trackBuff[k].len > minTrkLen)
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
        h_curvec[k * 2] = trackBuff[k].cur_frame_ptr->x;
        h_curvec[k * 2 + 1] = trackBuff[k].cur_frame_ptr->y;
        //std::cout<<h_curvec[k * 2]<<","<<h_curvec[k * 2 + 1]<<std::endl;
    }
    std::cout<<"end filter static point"<<std::endl;

    if(h_gcount[curTrkingIdx]<1)updateFlag=true;
    if(ofvBuff.len>0)
    {
        std::cout<<"start neighbor search"<<std::endl;
        unsigned char * curnbor = d_neighbor+offsetidx*nFeatures*nFeatures;
        float * curCos = d_cosine+offsetidx*nFeatures*nFeatures;
        float * curVelo = d_velo+offsetidx*nFeatures*nFeatures*2;
        cudaMemset(curnbor,0,nFeatures*nFeatures);
        cudaMemset(curCos,0,nFeatures*nFeatures*sizeof(float));
        cudaMemset(curVelo,0,nFeatures*nFeatures*sizeof(float)*2);
        cudaMemset(d_ofvec, 0, nFeatures* sizeof(ofv));
        cudaMemcpy(d_ofvec, ofvBuff.data, ofvBuff.len*sizeof(ofv), cudaMemcpyHostToDevice);
        cudaMemset(d_distmat,0,nFeatures*nFeatures*sizeof(float));
        searchNeighborBox <<<ofvBuff.len, ofvBuff.len >>>(curnbor,curCos,curVelo,d_ofvec,d_distmat,offsetidx, nFeatures);
        //searchNeighborMap <<<ofvBuff.len, ofvBuff.len >>>(curnbor,curCos,curVelo,d_ofvec,d_distmat,d_persMap, nFeatures);
        std::cout<<"end neighbor search:"<<std::endl;
        cudaMemcpy(h_curnbor,curnbor,nFeatures*nFeatures,cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distmat,d_distmat,nFeatures*nFeatures*sizeof(float),cudaMemcpyDeviceToHost);

        knn();

        cudaMemcpy(curnbor,h_curnbor,nFeatures*nFeatures,cudaMemcpyHostToDevice);
        cudaMemset(d_neighborD,0,nFeatures*nFeatures);
        neighborD<<<nFeatures,nFeatures>>>(d_neighbor,d_cosine,d_velo,d_neighborD,offsetidx);
        memcpy(h_prelabel ,h_label,nFeatures*sizeof(int));
        pregroupN = groupN;
        cudaMemcpy(h_neighborD,d_neighborD,nFeatures*nFeatures,cudaMemcpyDeviceToHost);

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
        if(updateFlag)
        {
            memcpy(label_final,h_label,nFeatures*sizeof(int));
        }

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
                if(trackBuff[i].len>1)
                {
                    h_bbstats[gidx*4+2]+=trackBuff[i].cur_frame_ptr->x-trackBuff[i].getPtr(trackBuff[i].len-2)->x;
                    h_bbstats[gidx*4+3]+=trackBuff[i].cur_frame_ptr->y-trackBuff[i].getPtr(trackBuff[i].len-2)->y;
                }
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
                h_bbstats[i*4+2]=h_bbstats[i*4+2]/float(h_gcount[i]);
                h_bbstats[i*4+3]=h_bbstats[i*4+3]/float(h_gcount[i]);
                if(calPolyGon)convex_hull(setPts[i],cvxPts[i]);
            }
        }
        if(updateFlag)
        {
            bbTrkBUff.clear();
            if(maxgroupN>1)
            {
                for(int i=(curTrkingIdx+1)%maxgroupN;i!=curTrkingIdx;i=(i+1)%maxgroupN)
                {
                    if(h_gcount[i]>5)
                    {
                        curTrkingIdx=i;
                        break;
                    }
                }
            }
            memcpy(h_precom,h_com,nFeatures*2*sizeof(float));

            memset(h_bbstats,0,nFeatures*4*sizeof(float));
            for(int i=0;i<nFeatures;i++)
            {
                int gidx = label_final[i];
                if(gidx)
                {
                    float dx = trackBuff[i].cur_frame_ptr->x-h_com[gidx*2];
                    float dy = trackBuff[i].cur_frame_ptr->y-h_com[gidx*2+1];
                    h_bbstats[gidx*4]+= dx*dx;
                    h_bbstats[gidx*4+1]+=dy*dy;
                }
            }
            for(int i=1;i<=maxgroupN;i++)
            {
                if(h_gcount[i]>0)
                {
                    h_bbstats[i*4] =sqrt(h_bbstats[i*4]/float(h_gcount[i])),
                    h_bbstats[i*4+1] = sqrt(h_bbstats[i*4+1]/float(h_gcount[i]));
                    if(h_bbstats[i*4]<h_bbstats[i*4+1])h_bbstats[i*4]=h_bbstats[i*4+1];
                    float curLen = (phk*h_com[i*2+1]+phb)/2;
                    if(h_bbstats[i*4]<curLen)h_bbstats[i*4]=curLen;
                }
            }
        }

        pttmp.x=h_precom[curTrkingIdx*2];
        pttmp.y=h_precom[curTrkingIdx*2+1];
        bbTrkBUff.updateAFrame(&pttmp);
        for(int i=1;i<=maxgroupN;i++)
        {
            if(h_gcount[i]>0)
            {
                float devix = h_bbstats[i*4],deviy = h_bbstats[i*4+1];
                devix=h_bbstats[i*4];
                deviy=h_bbstats[i*4];
                h_precom[i*2]=h_bbstats[i*4+2]+h_precom[i*2];
                h_precom[i*2+1]=h_bbstats[i*4+3]+h_precom[i*2+1];
                h_bbox[i*4]=h_precom[i*2]-devix,
                h_bbox[i*4+1]=h_precom[i*2+1]-deviy,
                h_bbox[i*4+2]=devix*2,
                h_bbox[i*4+3]=deviy*2;
                h_bbstats[i*4+2]=0;
                h_bbstats[i*4+3]=0;
            }
        }
        offsetidx=(offsetidx+1)%TIMESPAN;
    }
    std::cout<<"end frame"<<std::endl;
//    std::cout<<"gpuPrePts.size():"<<gpuPrePts.cols<<std::endl;
//    std::cout<<"nextPts.step():"<<nextPts.step<<"cols:"<<nextPts.cols<<"rows"<<nextPts.rows<<std::endl;

//    std::cout<<(intptr_t)gpuPrePts.data<<std::endl;
    gpuPrePts.upload(nextPts);
    //cudaMemcpy(gpuPrePts.data,nextPts.data,nFeatures*sizeof(float)*2,cudaMemcpyHostToDevice);
    return 1;
}
void KLTsparse_CUDA::knn()
{
    for(int i=0;i<ofvBuff.len;i++)
    {
        int ridx = ofvBuff.getPtr(i)->idx;
        int maxidx=0;
        for(int k=0;k<KnnK;k++)
        {
            h_KnnIdx[k]=-1;
        }
        for(int j=0;j<nFeatures;j++)
        {
            if(h_curnbor[ridx*nFeatures+j])
            {
                float val = h_distmat[ridx*nFeatures+j];
                if(h_KnnIdx[maxidx]<0|| val<h_distmat[ridx*nFeatures+h_KnnIdx[maxidx]])
                {
                    h_KnnIdx[maxidx]=j;
                    int maxi=0;
                    for(int k=0;k<KnnK;k++)
                    {
                        if(h_KnnIdx[k]<0)
                        {
                            maxi=k;
                            break;
                        }
                        else if(h_distmat[ridx*nFeatures+h_KnnIdx[k]]>h_distmat[ridx*nFeatures+h_KnnIdx[maxi]])
                        {
                                maxi=k;
                        }
                    }
                    maxidx=maxi;
                }
            }
        }
        memset(h_curnbor+ridx*nFeatures,0,nFeatures);
        for(int k=0;k<KnnK;k++)
        {
            if(h_KnnIdx[k]>=0)
            {
                h_curnbor[ridx*nFeatures+h_KnnIdx[k]]=1;
            }
        }
    }
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
        tmpn[i]=(h_neighborD[idx*nFeatures+i]);
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
                    for(int i=0;i<nFeatures;i++)
                    {
                        if(h_neighborD[idx*nFeatures+i])
                        {
                            nc++;
                            //if(tmpn[i])nnc++;
                            nnc+=(tmpn[i]>0);
                        }
                    }
                    if(nnc>nc*coNBThresh)
                    {
                        gcount++;
                        h_label[idx]=curK;
                        for(int i=0;i<nFeatures;i++)
                        {
                            tmpn[i]+=h_neighborD[idx*nFeatures+i];
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
                tmpn[i]+=h_neighborD[idx*nFeatures+i];
            }
            items[pos]=0;
        }
    }
    for(int i=0;i<nFeatures;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }

}
