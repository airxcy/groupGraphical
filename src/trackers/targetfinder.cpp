

#include "trackers/targetfinder.h"
#include "cuda.h"
#include "cuda_runtime.h"
//#include "opencv2/core/cuda_devptrs.hpp"
//#include "opencv2/gpu/device/common.hpp"
//#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::gpu::device;

int findCorners_gpu(PtrStepSzf eig, float threshold, PtrStepSzb mask, float2* corners, int max_count);
void sortCorners_gpu(PtrStepSzf eig, float2* corners, int count);
TargetFinder::TargetFinder(int maxCorners_, double qualityLevel_, double minDistance_,
        int blockSize_, bool useHarrisDetector_, double harrisK_)
{
    maxCorners = maxCorners_;
    qualityLevel = qualityLevel_;
    minDistance = minDistance_;
    blockSize = blockSize_;
    useHarrisDetector = useHarrisDetector_;
    harrisK = harrisK_;
}
void TargetFinder::setPersMat(GpuMat& m,int w,int h)
{
    fw=w,fh=h;
    m.download(cpuPersMap);
    cpuPersMap=cpuPersMap/10-1;
    cpuPersMap.convertTo(rangeMat,CV_8UC1);
    float* tmpptr=(float*)malloc(fw*fh*2*sizeof(float));
    tmpMat=Mat(1, fw*fh, CV_32FC2,tmpptr);
    tmp2.init(2,1024);
    m.copyTo(persMap);
    gpu::pow(persMap,0.05,persMap);
    double minVal, maxVal;
    gpu::minMax(persMap, & minVal, &maxVal );
    gpu::subtract(persMap,minVal*0.9,persMap);

    //gpu::add(persMap,2,persMap);
}
void TargetFinder::operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask)
{

    ensureSizeIsEnough(image.size(), CV_32F, eig_);
    cornerMinEigenVal(image, eig_, Dx_, Dy_, buf_, blockSize, 3);
    if(persMap.data)
    {
        //divide(eig_, persMap, eig_, 1);
        //gpu::add(eig_, 1, eig_);
    }
    double maxVal = 0;
    minMax(eig_, 0, &maxVal, GpuMat(), minMaxbuf_);

    ensureSizeIsEnough(1, std::max(1000, static_cast<int>(image.size().area() * 0.05)), CV_32FC2, tmpCorners_);

    int total = findCorners_gpu(eig_, static_cast<float>(maxVal * qualityLevel), mask, tmpCorners_.ptr<float2>(), tmpCorners_.cols);

    if (total == 0)
    {
        //corners.release();
        return;
    }

    sortCorners_gpu(eig_, tmpCorners_.ptr<float2>(), total);
    if (rangeMat.empty())
        tmpCorners_.colRange(0, maxCorners > 0 ? std::min(maxCorners, total) : total).copyTo(corners);
    else
    {
        tmpCorners_.colRange(0, total).download(tmpMat);
        float* tmpptr=(float*)tmpMat.data;
        tmp2.clear();
        float fp2[2];
        for(int i=0;i<total;i++)
        {
            int x =*tmpptr,y=*(tmpptr+1);
            memcpy(fp2,tmpptr,2*sizeof(float));
            tmpptr+=2;
            uchar range=rangeMat.at<uchar>(y,x);
            bool good=true;
            float* ptr2=tmp2.data;
            for(int j=0;j<tmp2.len;j++)
            {
                ptr2+=2;
                int x1=*(ptr2),y1=*(ptr2+1);
                int dx=abs(x1-x),dy=abs(y1-y);
                if(dx<range&&dy<range)
                {
                    good=false;
                    break;
                }
            }
            if(good)
            {
                tmp2.updateAFrame(fp2);
                if(tmp2.len==maxCorners)
                    break;
            }
        }
        corners.upload(Mat(1, tmp2.len, CV_32FC2,tmp2.data));
    }
    /*
    if (minDistance < 1)
        tmpCorners_.colRange(0, maxCorners > 0 ? std::min(maxCorners, total) : total).copyTo(corners);
    else
    {
        vector<Point2f> tmp(total);
        Mat tmpMat(1, total, CV_32FC2, (void*)&tmp[0]);
        tmpCorners_.colRange(0, total).download(tmpMat);

        vector<Point2f> tmp2;
        tmp2.reserve(total);

        const int cell_size = cvRound(minDistance);
        const int grid_width = (image.cols + cell_size - 1) / cell_size;
        const int grid_height = (image.rows + cell_size - 1) / cell_size;

        std::vector< std::vector<Point2f> > grid(grid_width * grid_height);

        for (int i = 0; i < total; ++i)
        {
            Point2f p = tmp[i];

            bool good = true;

            int x_cell = static_cast<int>(p.x / cell_size);
            int y_cell = static_cast<int>(p.y / cell_size);

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for (int yy = y1; yy <= y2; yy++)
            {
                for (int xx = x1; xx <= x2; xx++)
                {
                    vector<Point2f>& m = grid[yy * grid_width + xx];

                    if (!m.empty())
                    {
                        for(size_t j = 0; j < m.size(); j++)
                        {
                            float dx = p.x - m[j].x;
                            float dy = p.y - m[j].y;

                            if (dx * dx + dy * dy < minDistance * minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if(good)
            {
                grid[y_cell * grid_width + x_cell].push_back(p);

                tmp2.push_back(p);

                if (maxCorners > 0 && tmp2.size() == static_cast<size_t>(maxCorners))
                    break;
            }
        }

        corners.upload(Mat(1, static_cast<int>(tmp2.size()), CV_32FC2, &tmp2[0]));
    }
    */
}
