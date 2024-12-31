#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "MapDrawer.h"
#include "FrameHandle.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "MapLine.h"
#include "Map.h"
#include "Optimization.h"

void ReadPoseFile(std::string posePath, std::vector<cv::Mat>& Transform)
{
    // 打开csv文件
    std::ifstream file(posePath);

    // 读取每一行数据
    std::string line;
    while(std::getline(file, line))
    {
        // 将读取的每一行数据转换为字符串流
        std::stringstream ss(line);

        // 读取每一列数据
        std::string TimeStamp; std::getline(ss, TimeStamp, ' ');

        std::string tx; std::getline(ss, tx, ' ');
        std::string ty; std::getline(ss, ty, ' ');
        std::string tz; std::getline(ss, tz, ' ');
        
        std::string qx; std::getline(ss, qx, ' ');
        std::string qy; std::getline(ss, qy, ' ');
        std::string qz; std::getline(ss, qz, ' ');
        std::string qw; std::getline(ss, qw, ' '); // 注意读取顺序

        Eigen::Quaterniond quad(stod(qw), stod(qx), stod(qy), stod(qz)); // 注意构造顺序
        Eigen::Vector3d position(stod(tx), stod(ty), stod(tz));

        Eigen::Matrix3d RotationMatrix = quad.toRotationMatrix().transpose();
        Eigen::Vector3d TranslatVector = -RotationMatrix * position;

        cv::Mat Rcw;
        cv::eigen2cv(RotationMatrix, Rcw);

        cv::Mat tcw;
        cv::eigen2cv (TranslatVector, tcw);

        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(Tcw.rowRange(0,3).colRange(0, 3));
        tcw.copyTo(Tcw.rowRange(0,3).col(3));

        Transform.push_back(Tcw);
    }
    file.close();
}

void ReadImgPathFile(std::string filepath, std::vector<std::string>& imagepath)
{
    // 打开csv文件
    std::ifstream file(filepath);

    // 读取每一行数据
    std::string line;
    while(std::getline(file, line))
    {
        // 将读取的每一行数据转换为字符串流
        std::stringstream ss(line);

        // 读取数据并存储
        std::string path;
        std::getline(ss, path, ' ');

        imagepath.push_back(path);
    }
    file.close ();
}

void ReadPointLabelFile(std::string labelPath, std::vector<std::vector<cv::Point>>& pts_in_all_img)
{
    // 打开CSV文件
    std::ifstream file(labelPath);

    //速取每一行数据
    std::string line;
    while(std::getline(file, line))
    {
        // 将读取的每一行数据转换为字符串流
        std::stringstream ss(line);

        //读取每一列数据
        std::string TimeStamp;
        std::getline(ss, TimeStamp, ' ');

        std::vector<cv::Point> pts;
        while(1)
        {
            std::string x, y; 
            std::getline(ss, x, ' ');
            std::getline(ss, y, ' ');

            if(x.size() == 0)
                break;
            
            cv::Point pt(stod(x), stod(y)); 
            pts.push_back(pt);
        }
        pts_in_all_img.push_back(pts);
    }
    file.close();
}

void ReadLineLabelFile(std::string labelPath, std::vector<std::vector<cv::line_descriptor::KeyLine>>& lines_in_all_img)
{
    // 打开csv文件
    std::ifstream file(labelPath);

    // 读取每一行数据
    std::string line;
    while(std::getline(file, line))
    {
        // 将读取的每一行数据转换为宇符串流
        std::stringstream ss(line);

        // 读取每一列数据
        std::string TimeStamp;
        std::getline(ss, TimeStamp, ' ');

        std::vector<cv::line_descriptor::KeyLine> lines;
        while(1)
        {
            std::string sta_x, sta_y, end_x, end_y;
            std::getline(ss, sta_x, ' ');
            std::getline(ss, sta_y, ' ');
            std::getline(ss, end_x, ' ');
            std::getline(ss, end_y, ' ');

            if(sta_x.size() == 0)
                break;

            cv::line_descriptor::KeyLine line;
            line.startPointX = stod(sta_x); line.endPointX = stod(end_x);
            line.startPointY = stod(sta_y); line.endPointY = stod(end_y);
            lines.push_back(line);
        }
        lines_in_all_img.push_back(lines);
    }
    file.close();
}

// 用预先标注的点测试
void draw_labeled_Features(std::vector<KeyFrame*> kfs)
{
    std::vector<std::vector<cv::Point>> pts_label;
    ReadPointLabelFile("../data/pointlabel.txt", pts_label);

    std::vector<std::vector<cv::line_descriptor::KeyLine>> lines_label;
    ReadLineLabelFile("../data/linelabel.txt", lines_label);

    for (int i=0; i<kfs.size(); i++)
    {
        for (int j=0; j<pts_label[0].size(); j++)
        { 
            kfs[i]->InsertKeyPoint(pts_label[i][j]);
            kfs[i]->DrawKeyPoint(j);
        }
        
        for (int k=0; k<lines_label[0].size(); k++)
        {
            kfs[i]->InsertKeyLine(lines_label[i][k]);
            kfs[i]->DrawKeyLine(k);
        }
    }
}

cv::Mat skew_symmetric(const cv::Mat& vec)
{
    cv::Mat skew = cv::Mat::zeros(3, 3, CV_32F);
    skew.at<float>(0, 1) = -vec.at<float>(2);
    skew.at<float>(0, 2) =  vec.at<float>(1);
    skew.at<float>(1, 0) =  vec.at<float>(2);
    skew.at<float>(1, 2) = -vec.at<float>(0);
    skew.at<float>(2, 0) = -vec.at<float>(1);
    skew.at<float>(2, 1) =  vec.at<float>(0);

    return skew;
}

void test_Hcw_Hwc(cv::Mat Tcw)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3).clone();
    cv::Mat tcw = Tcw.rowRange(0,3).col(3).clone();
    cv::Mat Rwc = Rcw.t();
    cv::Mat twc = - Rwc * tcw;

    cv::Mat Hwc = cv::Mat::zeros(6, 6, CV_32F);
    Rwc.copyTo(Hwc.rowRange(0, 3).colRange(0, 3));
    Rwc.copyTo(Hwc.rowRange(3, 6).colRange(3, 6));

    cv::Mat tmp1 = - Rwc * skew_symmetric(tcw);
    tmp1.copyTo(Hwc.rowRange(0, 3).colRange (3, 6));

    cv::Mat Hcw = cv::Mat::zeros(6, 6, CV_32F);
    Rcw.copyTo(Hcw.rowRange(0, 3).colRange(0, 3));
    Rcw.copyTo(Hcw.rowRange(3, 6).colRange(3, 6));

    cv::Mat tmp2 = skew_symmetric(tcw) * Rcw;
    tmp2.copyTo(Hcw.rowRange(0, 3).colRange(3, 6));

    std::cout << Hwc * Hcw << std::endl;
}

int main(int argc, char **argv)
{
    // 读取外参：Tcw
    std::vector<cv::Mat> Tcw;
    ReadPoseFile("../data/data.csv", Tcw);

    // 读取图像 EuRoC: V1_01
    std::vector<std::string> imagePath;
    ReadImgPathFile("../data/imagepath.txt", imagePath);

    //读取相机内参
    std::string strSettingsFile = "../data/EuRoC.yaml";

    // 构建关键帧
    std::vector<KeyFrame*> kfs;
    for (int i=0; i<Tcw.size(); i++)
    {
        KeyFrame* kf = new KeyFrame(imagePath[i], Tcw[i], strSettingsFile, i);
        kfs.push_back(kf);
    }

    // 读取预先标注的点线数据
    draw_labeled_Features(kfs);

    // 关键帧插入地图
    Map* pMap = new Map();
    for (int i=0; i<kfs.size(); i++) 
        pMap->InsertKeyFrame(kfs[i]);
    
    // 显示图片
    for (int i=0; i<kfs.size(); i++)
    {
        cv::namedWindow("Frame " + std::to_string(kfs[i]->GetIndex()));
        cv::setMouseCallback("Frame "+ std::to_string(kfs[i]->GetIndex()), MouseOnImgHandle, (void*)kfs[i]);
    }

    while(1)
    {
        for(int i=0; i<kfs.size(); i++)
            cv::imshow("Frame " + std::to_string(kfs[i]->GetIndex()), kfs[i]->GetImage());

        int key = cv::waitKey(10);
        if(key == 32) // 按下 Space 键保存图片并退出
        {
            for(int i=0; i<kfs.size(); i++)
                cv::imwrite("./saveimg/Frame " + std::to_string(kfs[i]->GetIndex()) + ".png", kfs[i]->GetImage());
            std::cout << std::endl << "Save images success" << std::endl << std::endl;
            
            cv::destroyAllWindows();
            break;
        }
        
        if (key == 113) // 按下 Q 键退出
        {
            cv::destroyAllWindows();
            break;
        }
    }

    // 三角化 Mappoint, Mapline
    KeyFrame* kf1 = kfs[7];
    KeyFrame* kf2 = kfs[9];

    for(int i=0; i < kf1->GetKeyPoints()->size(); i++)
    {
        // 通过1、2帧图像重建三维空间点
        MapPoint* mp = new MapPoint(kf1, kf2, i, i);
        pMap->InsertMapPoint(mp);
    }

    for(int i = 0; i < kf1->GetKeyLines()->size(); i++)
    {
        // 通过1、2帧图像重建三维空间直线
        MapLine* ml = new MapLine(kf1, kf2, i, i, BY_TWOPLANES);
        // MapLine* ml = new MapLine(kf1, kf2, i, i, BY_ENDPOINTS);
        pMap->InsertMapLine(ml);
    }

    MapDrawer* pMapDrawer = new MapDrawer(pMap, strSettingsFile);
    pMapDrawer->SetCurrentCameraPose(kf1->GetPose());

    Viewer* pViewer = new Viewer(pMapDrawer, strSettingsFile);

    std::thread runthread([&]() {  // Start in new thread
    while(1)
    {
        if(pViewer->isFinished())
            break;

        if(pViewer->isRequestOptimize())
        {
            Optimizer* optimizer = new Optimizer(pMap);
            optimizer->onlyLineOpt();
            optimizer->onlyPointOpt();

            pViewer->SetOptimizeFinish();
        }
        usleep(5000);
    }
    }); // End the thread

    // Start the visualization thread
    pViewer->Run();

    runthread.join();

    return 0;
}