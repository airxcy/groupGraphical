#include "Qts/streamthread.h"

#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "trackers/utils.h"
#include <iostream>
#include <fstream>
//#include <stdlib.h>

//#include <direct.h>
#include "Qts/mainwindow.h"
#include <opencv2/opencv.hpp>
#include <ctime>

#include <QMessageBox>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QStringList>

using namespace cv;

VideoCapture cap;
Mat frame;
float fps=0;
char strbuff[100];
QDir qdirtmp;
Mat gray,segMat;

VideoWriter vwriter;
QImage rgbimg,img;
QPainter* painter;
Mat renderframe;

StreamThread::StreamThread(QObject *parent) : QThread(parent)
{
    restart = false;
    abort = false;
    pause = false;
    paused=false;
    bufflen=0;
    trkscene=NULL;
    framebuff=NULL;
    inited=false;
	firsttime = true;
    persDone=false;
    outVid =false;
    tracker = new KLTsparse_CUDA();
}
StreamThread::~StreamThread()
{
    abort = true;
	vwriter.release();
    cv0.wakeAll();
    wait(3000);
    terminate();
}

bool StreamThread::init()
{
    restart=false,pause=false;
    bufflen=0;

    if(!cap.isOpened())
    {
        cap.open(vidfname);
        std::cout<<"reopened"<<std::endl;
    }
    cap.set(CV_CAP_PROP_POS_FRAMES,0);
    frameidx=0;
    cap>>frame;
    fps=0;
    delay=25;
    bufflen=delay+10;
    cvtColor(frame,frame,CV_BGR2RGB);
    framewidth=frame.size[1],frameheight=frame.size[0];
    //cvtColor(frame,gray,CV_BGR2GRAY);
    if(framebuff==NULL)
    {
        framebuff = new FrameBuff();
        framebuff->init(frame.elemSize(),framewidth,frameheight,bufflen);
    }
    else
        framebuff->clear();
	
    frameByteSize=frame.size[0]*frame.size[1]*frame.elemSize();
    framebuff->updateAFrame(frame.data);
    frameptr=framebuff->cur_frame_ptr;
	parsefname();



    std::cout<<"persDone:"<<persDone<<std::endl;

	if (firsttime){
        tracker->init( framewidth, frameheight,frame.data,1024);
        //setUpPersBox();
        setUpPersMap();
        setUpROI();
        setupSegNeg();
    }
    if(!persDone)
    {
        pause=true;
    }
    if(outVid&&!vwriter.isOpened())
    {
        vwriter.open(vidid.toStdString() + "out.avi", CV_FOURCC('D','I','V','X'), 25, Size(framewidth, frameheight));
        rgbimg = QImage(framewidth, frameheight, QImage::Format_RGB888);
        img = QImage(framewidth, frameheight, QImage::Format_RGB888);
        renderframe = Mat(frameheight, framewidth,CV_8UC3);
        painter = new QPainter(&img);
    }
    inited=true;
	firsttime = false;
    return cap.isOpened();
}
void StreamThread::writeVid()
{
    painter->begin(&img);
    trkscene->render(painter);
    painter->end();
    rgbimg = img.convertToFormat(QImage::Format_RGB888);
    //std::cout<<rgbimg.byteCount()<<std::endl;
    memcpy(renderframe.data, rgbimg.bits(), rgbimg.byteCount());
    cvtColor(renderframe, renderframe, CV_BGR2RGB);
    vwriter << renderframe;
}
void StreamThread::parsefname()
{
	QFileInfo qvidfileinfo(vidfname.data());
	baseDirname = qvidfileinfo.path();
	vidid = qvidfileinfo.baseName();
	vidid = vidid + "_" + qvidfileinfo.completeSuffix();
	gtdir = baseDirname + "/" + vidid + "/";
	qdirstr = baseDirname + "/" + vidid + "/";
}
void StreamThread::setUpPersMap()
{
    if(!persDone)
    {
        QString savefname = baseDirname +"/"+vidid +"_persMap.csv";
        QFile qinfile(savefname);
        persMap=(float*)zalloc(frameheight*framewidth,sizeof(float));
        memset(persMap,0,frameheight*framewidth*sizeof(float));
        if (qinfile.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            std::cout<<savefname.toStdString()<<std::endl;
            QTextStream qinstream(&qinfile);
            QString line =qinstream.readLine();
            int linecount=0;

            while(!line.isEmpty())
            {
                QStringList valVec=line.split(",");
                for(int i=0;i<valVec.length();i++)
                {
                    persMap[linecount*framewidth+i]=valVec[i].toFloat();
                }
                linecount++;
                line =qinstream.readLine();
            }
//            Mat persMat(frameheight,framewidth,CV_32FC1,persMap);
//            imshow("asdf",persMat/255);
            tracker->setUpPersMap(persMap);
            persDone=true;
        }
    }
}
void StreamThread::setUpROI()
{
    QString savefname = baseDirname +"/"+vidid +"_ROI3.csv";
    QFile qinfile(savefname);
    if (qinfile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        std::cout<<savefname.toStdString()<<std::endl;
        QTextStream qinstream(&qinfile);
        QString line =qinstream.readLine();
        int linecount=0;

        while(!line.isEmpty())
        {
            QStringList valVec=line.split(",");
            roivec.push_back(valVec[0].toFloat());
            roivec.push_back(valVec[1].toFloat());
            linecount++;
            line =qinstream.readLine();
        }

        tracker->updateROICPU(roivec.data(),linecount);
    }
}
void StreamThread::setupSegNeg()
{
    QString savefname = baseDirname +"/"+vidid +"_segNeg.csv";
    QFile qinfile(savefname);
    if (qinfile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        std::cout<<savefname.toStdString()<<std::endl;
        QTextStream qinstream(&qinfile);
        QString line =qinstream.readLine();
        int linecount=0;
        std::vector<float> segneg;
        while(!line.isEmpty())
        {
            QStringList valVec=line.split(",");
            segneg.push_back(valVec[0].toFloat());
            segneg.push_back(valVec[1].toFloat());
            linecount++;
            line =qinstream.readLine();
        }

        tracker->updateSegNeg(segneg.data(),linecount);
    }
}
void StreamThread::updateSegCPU()
{
    if(segMat.empty())
    {
        unsigned char* ptr = new unsigned char[framewidth*frameheight];
        segMat = Mat(frameheight,framewidth,CV_8UC1,ptr);
    }
    QString imgdir = baseDirname +"/"+vidid +"_seg/";
    sprintf(strbuff,"%08d.png\0",frameidx);
    std::string imgpath = imgdir.toStdString()+strbuff;
    Mat segFrame =  imread(imgpath);
    cvtColor(segFrame,segMat,CV_BGR2GRAY);
    tracker->updateSegCPU(segMat.data);
}
void StreamThread::streaming()
{
    forever
    {
        if (abort)
            break;
        if(init())
        {
            emit initSig();
            frameidx=0;
            int fcounter=0;
            std::clock_t start = std::clock();
            double duration;
            int bbfcounter=0;
            while(!frame.empty())
            {
				if (abort)
					break;
                if (pause)
                {
                    mutex.lock();
                    paused=true;
                    cv0.wait(&mutex);
                    paused=false;
                    mutex.unlock();
                }
                cap >> frame;

                if(frame.empty())
                    break;
                //cvtColor(frame,gray,CV_BGR2GRAY);
                cvtColor(frame,frame,CV_BGR2RGB);
                if(applyseg)
                {
                    updateSegCPU();
                    tracker->applyseg=true;
                }
                else
                {
                    tracker->applyseg=false;
                }
                tracker->updateFlag=true;
                tracker->updateAframe(frame.data, frameidx);
                framebuff->updateAFrame(frame.data);
                frameptr=framebuff->cur_frame_ptr;
                frameidx++;
                //video write
                trkscene->update();
                if(outVid)emit aframedone();

                fcounter++;
                duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
                if(duration>=1)
                {
                    fps=fcounter/duration;
                    start=std::clock() ;
                    fcounter=0;
                }
                //msleep(1000);
            }
            vwriter.release();
            break;
        }

        else
        {
            //emit debug( "init Failed");
        }
        trkscene->clear();
        inited=false;
    }
}
void StreamThread::run()
{
    streaming();
}

void StreamThread::streamStart(std::string & filename)
{
    QMutexLocker locker(&mutex);
    //QMessageBox::question(NULL, "Test", "msg",QMessageBox::Ok);
    if (!isRunning()) {
        vidfname=filename;
        start(InheritPriority);
    }
    else
    {
        restart=true;
    }
}
