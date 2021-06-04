#include "mainwindow.h"
#include "form2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <QApplication>

using namespace std;
using namespace cv;

//int lucas_kanade(QString);

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    //form2 f;
    //f.show();
    w.show();
    return a.exec();
}


