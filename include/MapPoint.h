#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Utility.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

class MapPoint
{ 
public:
    MapPoint(const cv::Mat &Pos, size_t mpid);
    MapPoint(KeyFrame* kf1, KeyFrame* kf2, size_t kpID1, size_t kpID2);

    cv::Mat GetWorldPos();
    void UpdateWorldPos(const Vector3d pos);

    // 重投影到某一帧
    cv::Mat ProjectToCamera(KeyFrame* kf);

private:
    void Triangulate(const cv::Mat &kp1, const cv::Mat &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

protected:
    cv::Mat mWorldPos = cv::Mat::zeros(3, 1, CV_32F);

    std::mutex mMutexMapPoint;
};

#endif // MAPPOINT_H