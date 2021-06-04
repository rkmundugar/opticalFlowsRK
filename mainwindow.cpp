#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QLineEdit>
#include <QProgressBar>
#include <iostream>
#include <thread>
#include <fstream>
#include <string>
//#include <main.cpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/optflow/pcaflow.hpp>
#include <iostream>
#include <sys/stat.h>
#include <filesystem>
#include <direct.h>
#include <QApplication>

//#include <opencv2/cudaoptflow.hpp>
#include <opencv2/optflow/rlofflow.hpp>
#include <opencv2/optflow/pcaflow.hpp>
//#include <opencv2/optflow/deepflow.hpp>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //QLineEdit le = new QLineEdit();
    //cv::Mat image = cv::imread("C:\\Users\\HP\\Pictures\\Camera Roll\\im.jpg", 1);
    //cv::namedWindow("My Image");
    //cv::imshow("My Image", image);

    /*QString path = "C:\\Users\\HP\\Videos\\phone camera vids\\vid2.mp4";

    lucas_kanade(path);*/
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select video", "D:\\170911118\\sem8\\compare_results" );
    ui->lineEdit->setText(file);
}

void MainWindow::on_pushButton_2_clicked()
{

    QString file = QFileDialog::getExistingDirectory(this, "Select Destination folder", "D:\\170911118\\sem8\\compare_results" );
    ui->lineEdit_2->setText(file);
}

void MainWindow::splitFrames(QString vid,QString dest){

    cv::VideoCapture cap(vid.toStdString());

    int max = (int) cap.get(cv::CAP_PROP_FRAME_COUNT);
    //std::cout<<max<<std::endl;


    for(int frameNum = 0; frameNum < cap.get(cv::CAP_PROP_FRAME_COUNT)/2;frameNum++)
        {
            cv::Mat frame;

            cap >> frame; // get the next frame from video
            cap>> frame;

            QString filepath = dest + "\\frame_" + QVariant(frameNum).toString() +".jpg";

            cv::imwrite(filepath.toStdString(),frame);

            //std::cout<<cap.get(cv::CAP_PROP_POS_FRAMES)<<std::endl;

            int per = (frameNum*200)/max;

            ui->progressBar->setValue(per);

        }
}

double store[10][1000];

//Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
//Ptr<DenseOpticalFlow> algo = optflow::createOptFlow_DualTVL1();
//Ptr< DenseOpticalFlow > pcaflow = makePtr< optflow::OpticalFlowPCAFlow >();

//Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
//Ptr< DenseOpticalFlow > pcaflow = makePtr< optflow::OpticalFlowPCAFlow >();
Ptr<DenseOpticalFlow> algo;
void MainWindow::on_pushButton_3_clicked()
{
    QString vid = ui->lineEdit->text();
    QString dest = ui->lineEdit_2->text();

   // splitFrames(vid,dest);
    bool to_gray = 1;
    //QString dest = "";
    int p1 = 8;
    int p2 = 128;
    //int store[10][1000];
    //Method method = "calcOpticalFlowSparseToDense";

    future = std::async(std::launch::async, [this,vid,dest]{
        splitFrames(vid,dest);
        emit frameSplitComp();
    });

        QString names[9];

        QString name = "Farneback";
        names[0]=name;
        dense_optical_flow(0,vid,dest,name,cv::calcOpticalFlowFarneback,to_gray, 0.5,3,15,3,5,1.2,0);


        to_gray = 0;
        name = "RLOF";
        names[1]=name;
        dense_optical_flow(
               1,vid,dest,name, cv::optflow::calcOpticalFlowDenseRLOF, to_gray,
               Ptr<cv::optflow::RLOFOpticalFlowParameter>(), 1.f, Size(6,6),
               cv::optflow::InterpolationType::INTERP_EPIC, 128, 0.05f, 999.0f,
               15, 100, true, 500.0f, 1.5f, false);


        name = "sparseToDense";
        names[2]=name;
        to_gray = 1;
        dense_optical_flow(2,vid,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);



        name = "PCAflow";
        names[3]=name;
        to_gray = 1;
        dense_optical_flow(3,vid,dest,name,optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);

        name="DISflow";
        names[4]=name;
        to_gray=1;
        dense_optical_flow(4,vid,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);

        name="deepflow";
        names[5]=name;
        to_gray=1;
        dense_optical_flow(5,vid,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);

        name="simpleflow";
        names[6]=name;
        to_gray=0;
        dense_optical_flow(6,vid,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);

        name = "TVL1_GPU";
        names[7]=name;
        to_gray = 1;
        dense_optical_flow(7,vid,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);

        name = "TVL1";
        names[8]=name;
        to_gray = 1;
        dense_optical_flow(8,vid,dest,name, optflow::calcOpticalFlowSparseToDense, to_gray, 8, 128, 0.05f, true, 500.0f, 1.5f);


        VideoCapture capture(samples::findFile(vid.toStdString()));
        for(int i=0;i<9;i++)
        {
            cout<<"*"<<i<<"* : ";
            for(int j=0;j<capture.get(cv::CAP_PROP_FRAME_COUNT)/2;j++)
            {
                cout<<store[i][j]<<" , ";
            }
            cout<<"\n";
        }


        std::ofstream myfile;
        myfile.open(dest.toStdString()+"//data.csv");

        for(int i=0;i<9;i++)
        {
            myfile<<names[i].toStdString()<<",";
            for(int j=0;j<capture.get(cv::CAP_PROP_FRAME_COUNT)/2;j++)
            {
                myfile<<store[i][j]<<",";
            }
            myfile<<"\n";
        }



}


int MainWindow::lucas_kanade(QString filename, QString dest)
{
    // Read the video
    cv::VideoCapture cap(filename.toStdString());
    if (!cap.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << std::endl;
        return 0;
    }

    // Create random colors
    vector<Scalar> colors;
    cv::RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r,g,b));
    }

    Mat old_frame, old_gray;
    std::vector<cv::Point2f> p0, p1;

    // Read first frame and find corners in it
    cap >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_RGB2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    // Create a mask image for drawing purposes
    cv::Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    int counter = 1;
    int max = (int) cap.get(cv::CAP_PROP_FRAME_COUNT);
    while(counter <= cap.get(cv::CAP_PROP_FRAME_COUNT)){
        // Read new frame
        Mat frame, frame_gray;
        cap >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // Calculate optical flow
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        if(cap.get(CAP_PROP_POS_FRAMES) < (max-1))
        {
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
        vector<Point2f> good_new;

        // Visualization part
        for(uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if(status[i] == 1) {
                good_new.push_back(p1[i]);
                // Draw the tracks
                line(mask,p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }

        p0 = good_new;
        }

        // Display the demo

        Mat img;
        add(frame, mask, img);
       // if (QMetaType::save) {
            //string save_path = "./optical_flow_frames/frame_" + to_string(counter) + ".jpg";
            //imwrite(save_path, img);
       // }

        QString filepath = dest + "\\frame_" + QVariant(counter).toString() +".jpg";
        cv::imwrite(filepath.toStdString(),frame);




        imshow("flow", img);
        int keyboard = waitKey(25);
        //if (keyboard == 'q' || keyboard == 27) break;
        //else continue;

        // Update the previous frame and previous points
        old_gray = frame_gray.clone();

        counter++;

        int per = (counter*100)/max;
        ui->progressBar->setValue(per);

}
    destroyWindow("flow");
}


template <typename Method, typename... Args>
void MainWindow::dense_optical_flow(int tag,QString filename, QString dest, QString name,Method method, bool to_gray, Args&&... args)
{


    // Read the video and first frame
    VideoCapture capture(samples::findFile(filename.toStdString()));
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
    }
    Mat frame1, prvss,prvs;
    capture >> frame1;

    //Preprocessing for exact method
    if (to_gray)
        cvtColor(frame1, prvss, COLOR_BGR2GRAY);
    else
        prvss = frame1;

    QString folder = dest + "\\"+name;
    int status = mkdir(folder.toStdString().c_str());

    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH)/2;
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT)/2;

    VideoWriter outputVideo;
    QString vname = folder + "\\"+name + "_video.mp4";

    Size size = Size(frame_width, frame_height);
    int codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
    VideoWriter video(vname.toStdString(), codec, 15, size, true);

    int counter = 1;
    int max = (int) capture.get(cv::CAP_PROP_FRAME_COUNT);
    while (counter <= capture.get(cv::CAP_PROP_FRAME_COUNT)) {
        auto start = chrono::steady_clock::now();
        // Read the next frame
        Mat frame2, nexts,next;

        for(int i=0;i<2;i++)
        capture >> frame2;

        if (frame2.empty())
            break;

        // Preprocessing for exact method
        if (to_gray)
            cvtColor(frame2, nexts, COLOR_BGR2GRAY);
        else
            nexts = frame2;

       cv::resize(prvss,prvs, Size(prvss.cols/2,prvss.rows/2));
       cv::resize(nexts,next, Size(nexts.cols/2,nexts.rows/2));

        Mat magnitude, angle, magn_norm;
        Mat _hsv[3], hsv, hsv8, bgr;
        Mat flow(prvs.size(), CV_32FC2);
        if(capture.get(CAP_PROP_POS_FRAMES) < (max-1))
        {
        // Calculate Optical Flow
            if(name == "DISflow")
            {
                algo = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
                algo->calc(prvs,next,flow);
            }
            else if(name == "PCAflow")
            {
                algo = makePtr< optflow::OpticalFlowPCAFlow >();
                algo->calc(prvs,next,flow);
            }
            else if(name == "TVL1")
            {
                //algo->calc(frame1,frame2,flow.getUMat(ACCESS_RW));
                algo = optflow::createOptFlow_DualTVL1();
                algo->calc(prvs,next,flow);
            }
            else if(name == "TVL1_GPU")
            {
                algo = optflow::createOptFlow_DualTVL1();
                algo->calc(prvs,next,flow.getUMat(ACCESS_RW));
             }
            else if(name == "deepflow")
            {
                algo = optflow::createOptFlow_DeepFlow();
                algo->calc(prvs, next, flow);
            }
            else if(name == "simpleflow")
            {
                //algo = optflow::createOptFlow_SimpleFlow();
                //algo->calc(prvs,next, flow);
                optflow::calcOpticalFlowSF(prvs, next,
                                    flow,
                                    3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
            }
            else{
        method(prvs, next, flow, std::forward<Args>(args)...);
            }
        //(prvs, next, flow, std::forward<Args>(args)...);
           // algorithm->calc(prvs,next,flow);
            //pcaflow->calc( prvs, next, flow );
            //algo->calc(prvs, next, flow);
            //algo->calc(prvs, next, flow.getUMat(ACCESS_RW));
        }

        // Visualization part
        Mat flow_parts[2];
        split(flow, flow_parts);

        // Convert the algorithm's output into Polar coordinates

        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));


        // Build hsv image

        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);


        // Display the results
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);

        Mat qim;

        cvtColor(bgr,qim,COLOR_BGR2RGB);

        auto end = chrono::steady_clock::now();
        auto diff = end-start;

        int keyboard = waitKey(5);       
        QString time = QString::number(chrono::duration <double, milli> (diff).count())+"ms";
        putText(qim,name.toStdString(),cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        putText(qim,time.toStdString(),cv::Point(50,80),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        QPixmap Q = QPixmap::fromImage(QImage(qim.data, qim.cols, qim.rows, qim.step, QImage::Format_RGB888));

        Mat ims;
        cv::resize(qim,ims, Size(qim.cols/2,qim.rows/2));
        imshow("flow", qim);

        store[tag][counter-1]=chrono::duration <double, milli> (diff).count();


        //namespace fs = std::filesystem;

        QString filepath = folder + "\\frame_" + QVariant(counter).toString() +".jpg";
        cv::imwrite(filepath.toStdString(),qim);

        video<<qim;


        // Update the previous frame
        prvss = nexts;
        counter++;

        int per = (counter*200)/max;
        ui->progressBar->setValue(per);
    }
    destroyWindow("flow");
    //destroyWindow("frame");
}


