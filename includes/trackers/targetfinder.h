#ifndef TARGETFINDER_H
#define TARGETFINDER_H

#include <vector>
#include <memory>
#include <iosfwd>

#include "opencv2/core/gpumat.hpp"
#include "trackers/buffers.h"
using namespace cv;
using namespace cv::gpu;
class TargetFinder
{
public:
    TargetFinder(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0,
        int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04);

    void setPersMat(GpuMat& m,int w,int h);
    void fetchPoints(int total,GpuMat& corners);
    //! return 1 rows matrix with CV_32FC2 type
    void operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask= GpuMat());

    int maxCorners;
    double qualityLevel;
    double minDistance;

    int blockSize;
    bool useHarrisDetector;
    double harrisK;

    void releaseMemory()
    {
        Dx_.release();
        Dy_.release();
        buf_.release();
        eig_.release();
        minMaxbuf_.release();
        tmpCorners_.release();
    }
    int fw,fh;
    Mat cpuPersMap;
    Mat rangeMat;
    Mat tmpMat;
    Buff<float> tmp2;
//private:
    GpuMat persMap;
    GpuMat Dx_;
    GpuMat Dy_;
    GpuMat buf_;
    GpuMat eig_;
    GpuMat minMaxbuf_;
    GpuMat tmpCorners_;
};
#endif // TARGETFINDER_H

