#include "Qts/mainwindow.h"
#include "Qts/streamthread.h"
#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"

//#include <direct.h>

#include <QtWidgets>
#include <QSizePolicy>
#include <iostream>
#include <QPalette>
#include <QKeySequence>
#include <QFontDatabase>
#include <QStringList>
//! [1]
char cbuff[200];
MainWindow::MainWindow()
{

    cWidget = new QWidget(this,Qt::FramelessWindowHint);
    setCentralWidget(cWidget);

    streamThd = new StreamThread(this);

    trkscene=NULL;
    setupLayout();
    makeConns();
    move(100, 0);

}
MainWindow::~MainWindow()
{
    streamThd->abort=true;
    streamThd->~StreamThread();
}
void MainWindow::setupLayout()
{
    /** views **/
    defaultscene = new DefaultScene(0, 0, 440, 240);
    gview = new GraphicsView(defaultscene,cWidget);
    gview->setFixedSize(defaultscene->width()+4,defaultscene->height()+4);
    gview->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    cWidget->setFixedSize(gview->minimumSize());
}
void MainWindow::makeConns()
{
    connect(defaultscene,SIGNAL(clicked(QGraphicsSceneMouseEvent *)),this,SLOT(gviewClicked(QGraphicsSceneMouseEvent *)));
    connect(streamThd,SIGNAL(initSig()),this,SLOT(initUI()),Qt::BlockingQueuedConnection);
    connect(streamThd,SIGNAL(aframedone()),this,SLOT(rendscene()),Qt::BlockingQueuedConnection);
}
void MainWindow::gviewClicked(QGraphicsSceneMouseEvent * event)
{
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open vid"), "/home/cyxia/globalCenter", tr("Vid Files (*.avi *.mp4 *.mkv *.MTS *.MOV)"));
    if(!fileName.isEmpty())
    {
        std::string tmpstr = fileName.toStdString();
        streamThd->streamStart(tmpstr);
    }
}

void MainWindow::initUI()
{
    int fw=streamThd->framewidth,fh=streamThd->frameheight;
    trkscene = new TrkScene(0, 0, fw, fh);
    trkscene->streamThd=streamThd;
    streamThd->trkscene=trkscene;
    trkscene->sdf.translate(fw,0);
    trkscene->rectbrush.setTransform(trkscene->sdf);
    gview->setFixedSize(fw+4,fh+4);
    gview->setScene(trkscene);
    trkscene->updateFptr(streamThd->frameptr,streamThd->frameidx);
    cWidget->setFixedSize(gview->minimumSize());
    showFullScreen();

}
void MainWindow::rendscene()
{
        streamThd->writeVid();
}
void MainWindow::keyPressEvent(QKeyEvent * event)
{
    if(event->key()==Qt::Key_Space)
    {
        if(streamThd)
        {
            if(streamThd->paused)
            {
            streamThd->pause=false;
            streamThd->cv0.wakeAll();
            }
            else
                streamThd->pause=true;
        }
    }
    if(event->key()==Qt::Key_1)
    {
        if(streamThd)
        {
            streamThd->tracker->render=!streamThd->tracker->render;
        }
    }
    if(event->key()==Qt::Key_2)
    {
        if(streamThd)
        {
            streamThd->applyseg=!streamThd->applyseg;
        }
    }
    if(event->key()==Qt::Key_3)
    {
        if(streamThd)
        {
            streamThd->tracker->groupOnFlag=!streamThd->tracker->groupOnFlag;
        }
    }
}
