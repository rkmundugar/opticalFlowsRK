#include "form2.h"
#include "ui_form2.h"
#include <QFileDialog>
#include <QLineEdit>
#include <QProgressBar>
#include <QWidget>
#include <iostream>
#include <thread>
#include <QRadioButton>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow.hpp>
//#include <opencv2/cudaoptflow.hpp>
#include <opencv2/optflow/rlofflow.hpp>
#include <opencv2/optflow/pcaflow.hpp>
#include <iostream>
#include <QApplication>

using namespace std;
using namespace cv;


form2::form2(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::form2)
{
    ui->setupUi(this);
}

form2::~form2()
{
    delete ui;
}

void form2::on_pushButton_2_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select frame 1", "D:\\" );
    ui->lineEdit->setText(file);

    Mat f1 = imread(file.toStdString());
    cvtColor(f1,f1,COLOR_BGR2RGB);
    putText(f1,"frame 1",cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,0,0),2,false);
    QPixmap Q = QPixmap::fromImage(QImage(f1.data, f1.cols, f1.rows, f1.step, QImage::Format_RGB888));
    ui->label_2->setPixmap(Q.scaled(ui->label_2->size(),Qt::KeepAspectRatio, Qt::SmoothTransformation));
}



void form2::on_pushButton_3_clicked()
{
    //QString file = QFileDialog::getExistingDirectory(this, "Select Destination folder", "D:\\" );
    //ui->lineEdit_2->setText(file);
}

void form2::on_pushButton_4_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select frame 2", "D:\\" );
    ui->lineEdit_3->setText(file);

    Mat f2 = imread(file.toStdString());
    cvtColor(f2,f2,COLOR_BGR2RGB);

    putText(f2,"frame 2",cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,0,0),2,false);
    QPixmap Q = QPixmap::fromImage(QImage(f2.data, f2.cols, f2.rows, f2.step, QImage::Format_RGB888));
    ui->label_8->setPixmap(Q.scaled(ui->label_8->size(),Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

//template <typename Method>
//Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
//Ptr< DenseOpticalFlow > pcaflow = makePtr< optflow::OpticalFlowPCAFlow >();
//Ptr<DenseOpticalFlow> algo = optflow::createOptFlow_DualTVL1();


void form2::on_pushButton_clicked()
{
    QString frame1 = ui->lineEdit->text();
    QString frame2 = ui->lineEdit_3->text();
    QString dest = "";
    bool to_gray;


        to_gray=1;
        Mat fb;
        QString name = "Farneback";
        destroyWindow("flow");
        future = async(launch::async, [this,frame1,frame2,name,dest,to_gray,fb]{
            Mat fb = dense_optical_flow(frame1,frame2,dest,name,cv::calcOpticalFlowFarneback,to_gray, 0.5,3,15,3,5,1.2,0);
            emit frameSplitComp();
        });


        destroyWindow("flow");
        name = "LukasKanade";
        to_gray=1;
        future = async(launch::async, [this,frame1,frame2,dest,name,to_gray]{
             Mat lk = dense_optical_flow(frame1,frame2,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);
             emit frameSplitComp();
        });


        destroyWindow("flow");
        name = "DISflow";
        to_gray=1;
        future = async(launch::async, [this,frame1,frame2,dest,name,to_gray]{
            Mat dis =  dense_optical_flow(frame1,frame2,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);
            emit frameSplitComp();
        });

        destroyWindow("flow");
        name = "PCAflow";
        to_gray=1;
        future = async(launch::async, [this,frame1,frame2,dest,name,to_gray]{
            Mat dis =  dense_optical_flow(frame1,frame2,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);
            emit frameSplitComp();
        });

        destroyWindow("flow");
        name = "TVL1";
        to_gray=1;
        future = async(launch::async, [this,frame1,frame2,dest,name,to_gray]{
            Mat dis =  dense_optical_flow(frame1,frame2,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);
            emit frameSplitComp();
        });


        to_gray = 0;
        name = "RLOF";
        destroyWindow("flow");
        future = async(launch::async, [this,frame1,frame2,dest,name,to_gray]{
            Mat rlof = dense_optical_flow(
                   frame1,frame2,dest,name, cv::optflow::calcOpticalFlowDenseRLOF, to_gray,
                   Ptr<cv::optflow::RLOFOpticalFlowParameter>(), 1.f, Size(6,6),
                   cv::optflow::InterpolationType::INTERP_EPIC, 128, 0.05f, 999.0f,
                   15, 100, true, 500.0f, 1.5f, false);
            emit frameSplitComp();
        });

}



template <typename Method, typename... Args>
Mat form2::dense_optical_flow(QString filename,QString filename2, QString dest,QString name, Method method, bool to_gray, Args&&... args)
{
    auto start = chrono::steady_clock::now();

    Mat frame1, frame2;
    if(to_gray==0){
    frame2 = imread(filename2.toStdString());
    frame1 = imread(filename.toStdString());
    }
    else
    {
    frame2 = imread(filename2.toStdString(),IMREAD_GRAYSCALE);
    frame1 = imread(filename.toStdString(),IMREAD_GRAYSCALE);
    }


        Mat magnitude, angle, magn_norm;
        Mat _hsv[3], hsv, hsv8, bgr;
        Mat flow(frame1.size(), CV_32FC2);

        if(name == "DISflow")
        {
            //algorithm->calc(frame1,frame2,flow);
        }
        else if(name == "PCAflow")
        {
            //pcaflow->calc(frame1,frame2,flow);
        }
        else if(name == "TVL1")
        {
            //algo->calc(frame1,frame2,flow.getUMat(ACCESS_RW));
            //algo->calc(frame1,frame2,flow);
        }
        else
        {
            method(frame1, frame2, flow, std::forward<Args>(args)...);
        }
        Mat flow_parts[2];
        split(flow, flow_parts);



        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));



        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);


        cvtColor(hsv8, bgr, COLOR_HSV2BGR);

        //imshow(name.toStdString(), bgr);

        //ui->graphicsView->;

        Mat qim;

        cvtColor(bgr,qim,COLOR_BGR2RGB);
        auto end = chrono::steady_clock::now();

        auto diff = end-start;

        QString time = QString::number(chrono::duration <double, milli> (diff).count())+"ms";
        putText(qim,name.toStdString(),cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        putText(qim,time.toStdString(),cv::Point(50,80),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        QPixmap Q = QPixmap::fromImage(QImage(qim.data, qim.cols, qim.rows, qim.step, QImage::Format_RGB888));

        if(name == "TVL1")
        {
            ui->label_4->setPixmap(Q.scaled(ui->label_4->size(),Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        else if(name == "RLOF")
        {
            ui->label_5->setPixmap(Q.scaled(ui->label_4->size(),Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        else if (name == "PCAflow")
        {
            ui->label_6->setPixmap(Q.scaled(ui->label_4->size(),Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        else if(name == "DISflow")
        {
            ui->label_7->setPixmap(Q.scaled(ui->label_4->size(),Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }

        //cout<<chrono::duration <double, milli> (diff).count() << " ms" << endl;


        //QString filepath = dest + "\\frame_" +name+".jpg";
        //cv::imwrite(filepath.toStdString(),bgr);
        waitKey(0);



        return qim;


}



