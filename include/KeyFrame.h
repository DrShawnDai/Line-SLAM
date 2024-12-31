#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "LBDextractor.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <string>
#include <unistd.h>
#include <math.h>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

using namespace std;

class KeyFrame
{ 
public:
    KeyFrame(const string imagePath, const cv::Mat &Pos, const string strSettingsFile, const size_t id);

    size_t GetIndex();

    cv::Mat GetImage();

    // 位姿相关函数
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // 关键角点相关函数
    void InsertKeyPoint(const cv::Point &point); // 手动插入

    cv::Point* GetKeyPoint(const size_t id); 
    std::vector<cv::Point>* GetKeyPoints();

    void DrawKeyPoint (const size_t id);
    void DrawPoint(const cv::Point pt, const size_t id, cv::Scalar color = cv::Scalar(255, 0, 0));
    
    // 关键线特征提取
    void ExtractLBD();
    void SetExtractor(LBDextractor* lbd_extract);

    // 关键线特征相关函数
    void InsertKeyLine(const cv::line_descriptor::KeyLine &line); // 手动插入

    cv::line_descriptor::KeyLine* GetKeyLine(const size_t id) ; 
    std::vector<cv::line_descriptor::KeyLine>* GetKeyLines();

    cv::Mat GetLineDescriptor(const size_t id);
    cv::Mat GetAllLinesDescriptors();

    void AssignLineFeaturesToGrid();
    bool PosInGrid(const cv:: line_descriptor::KeyLine &kl, int &posX, int &posY);
    std::vector<size_t> GetLineFeaturesInArea(const float &x, const float &y, const float &r) const;

    void DrawKeyLine(const size_t id);
    void DrawLine(const cv::line_descriptor::KeyLine line, cv::Scalar color = cv::Scalar(255, 255, 0), const int width = 1);

    // 将像素坐标转换至相机归一化坐标
    cv::Mat pixelToCam(const cv::Mat &pixel);

    // 将归一化坐标转至世界坐标
    cv::Mat camToWorld(const cv::Mat &Pc);

    // Initial settings (computed once)
    static bool mbInitialComputations;

    static cv::Mat K, K_inv;

    static cv::Mat DistCoef;

    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;

    std::vector<std::size_t> mLineGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

protected:
    size_t index;

    cv::Mat image;
    cv::Mat undist_image;

    cv::Mat Tcw, Twc;

    std::vector<cv::Point> keypoints;

    std::vector<cv::line_descriptor::KeyLine> mvKeyLines;
    cv::Mat mLineDescriptors;

    LBDextractor* mpLBDextractor;
};

#endif // KEYFRAME_H