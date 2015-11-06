#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>
#include <stdio.h>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
QPointF points[100];

void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}
void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //std::cout<<streamThd->inited<<std::endl;
    if(streamThd!=NULL&&streamThd->inited)
    {
        updateFptr(streamThd->frameptr, streamThd->frameidx);
    }
    painter->setBrush(bgBrush);
    painter->drawRect(rect);
    painter->setBrush(QColor(0,0,0,50));
    painter->drawRect(rect);
    if(streamThd!=NULL&&streamThd->inited)
    {
        painter->setPen(Qt::red);
        painter->setFont(QFont("System",10,2));
        std::vector<FeatBuff>& klttrkvec=streamThd->tracker->trackBuff;

        int nFeatures= streamThd->tracker->nFeatures;
        int* labelvec=streamThd->tracker->label_final;
        unsigned char* clrvec=streamThd->tracker->h_clrvec;
        unsigned char* neighborD = streamThd->tracker->h_neighbor;
        int groupN = streamThd->tracker->groupN;
        float* groupvec = streamThd->tracker->h_com;
        int nSearch=streamThd->tracker->nSearch;
        float* persmap=streamThd->tracker->h_persMap;
        painter->drawText(rect, Qt::AlignLeft|Qt::AlignTop,QString::number(streamThd->fps)+","+QString::number(groupN));
        std::vector<float> & roivec = streamThd->roivec;
        double x0,y0,x1,y1;

        for(int i=0;i<klttrkvec.size();i++)
        {
            FeatBuff& klttrk= klttrkvec[i];
            int label = labelvec[i];
            unsigned char r=255,g=255,b=255;

            if(label)
            {
                r=clrvec[label*3],g=clrvec[label*3+1],b=clrvec[label*3+2];
                x1=klttrk.cur_frame_ptr->x,y1=klttrk.cur_frame_ptr->y;
                linepen.setColor(QColor(r, g, b,70));
                linepen.setWidth(1);
                painter->setPen(linepen);

                for (int j = i+1; j < nFeatures; j++)
                {
                    int xj = klttrkvec[j].cur_frame_ptr->x, yj = klttrkvec[j].cur_frame_ptr->y;
                    int label2=labelvec[j];
                    if (neighborD[i*nFeatures+j]&&label2==label)//&&abs(xj-x1)+abs(y1-yj)<100
                    {
                        painter->drawLine(x1, y1, xj, yj);
                    }
                }
                linepen.setColor(QColor(0, 0, 0,255));
                painter->setPen(linepen);
                painter->drawText(x1,y1,QString::number(klttrk.len));

            }

            linepen.setWidth(1);
            int startidx=std::max(1,klttrk.len-15);
            for(int j=startidx;j<klttrk.len;j++)
            {

                x1=klttrk.getPtr(j)->x,y1=klttrk.getPtr(j)->y;
                x0=klttrk.getPtr(j-1)->x,y0=klttrk.getPtr(j-1)->y;
                int denseval = ((j - startidx)/(15+0.1))*100;
                int indcator = (denseval) > 255;
                int alpha = indcator * 255 + (1 - indcator)*(denseval);
                linepen.setColor(QColor(r, g, b,alpha));
                painter->setPen(linepen);
                painter->drawLine(x0,y0,x1,y1);
            }

        }
        linepen.setWidth(2);
        linepen.setColor(QColor(255, 255, 255,255));
        painter->setPen(linepen);


        for(int i=0;i<groupN;i++)
        {

            int x =groupvec[2*i]+0.5,y=groupvec[2*i+1]+0.5;
            float persval = persmap[y*streamThd->framewidth+x];
            painter->setFont(QFont("System",persval/10+10,2));
            painter->drawText(x,y,QString::number(i));
        }


        for(int i=0;i<nSearch;i++)
        {
            linepen.setColor(QColor(255,0,0));
            linepen.setWidth(2);
            cv::Vec2f cnrp = streamThd->tracker->corners.at<cv::Vec2f>(i);
            int x=cnrp[0],y=cnrp[1];
            float val = (i+0.1)/(nSearch+0.1)*255.0;
            //std::cout<<val<<std::endl;
            linepen.setColor(QColor(val,255-val,0));
            painter->setPen(linepen);
            painter->drawPoint(x,y);
            //painter->drawText(x,y,QString::number(i));
            //painter->drawLine(x-10,y,x+10,y);
            //painter->drawLine(x,y-10,x,y+10);
        }



    }

    //update();
    //views().at(0)->update();
}
void TrkScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        int x = event->scenePos().x(),y=event->scenePos().y();
        DragBBox* newbb = new DragBBox(x-10,y-10,x+10,y+10);
        int pid = dragbbvec.size();
        newbb->bbid=pid;
        newbb->setClr(255,255,255);
        sprintf(newbb->txt,"%c\0",pid+'A');
        dragbbvec.push_back(newbb);
        addItem(newbb);
    }
    QGraphicsScene::mousePressEvent(event);
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //std::cout<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
