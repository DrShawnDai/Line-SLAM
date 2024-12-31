#ifndef MAPLINE_H 
#define MAPLINE_H

#include "KeyFrame.h"
#include "Utility.h"

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

enum ConstuctType { BY_ENDPOINTS = 0, BY_TWOPLANES = 1 };

class MapLine
{ 
public:
    MapLine(const cv::Mat &Plucker, const cv::Mat &Points);
    MapLine(KeyFrame* kf1, KeyFrame* kf2, size_t lineID1, size_t lineID2, ConstuctType type);

    cv::Mat GetNormal(); 
    cv::Mat GetDirect();
    cv::Mat GetOrthonormal();
    cv::Mat GetEndPoints();

    void UpdateByOrth(const Vector4d orth);

    // normal 投影到相机平面
    cv::Mat ProjectToCamera(KeyFrame* kf);

private:
    void Triangulate(const cv::Mat &kp1, const cv::Mat &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    // 通过三点确定一平面
    // Ax+By+Cz+D=0，过三个点 P1，P2，P3，解三元一次方程组
    cv::Mat reconstructPlane(const cv::Mat &P1, const cv::Mat &P2, const cv::Mat &P3);

    // 通过"两平面交线"确定空间直线的普吕克矩阵
    cv::Mat PluckerFromTwoPlanes (const cv::Mat &plane1, const cv::Mat &plane2);
    
    // 通过"两个端点"确定空间直线的普吕克矩阵
    cv::Mat PluckerFromTwoPoints(const cv::Mat &x3D1, const cv::Mat &x3D2);

    // 普吕克矩阵转化为普吕克坐标
    cv::Mat PluckerMatToCoordinate(const cv::Mat &plucker_mat); 
    cv::Mat PluckerCoordinateToMat(const cv::Mat &plucker_coord);
    
    // 普吕克坐标转化为正交表示 (Orthonormal)
    void ComputeOrthByPlucker(); 
    void ComputePluckerByOrth();
    
    // 根据优化后的正交表示orth更新端点
    void UpdateEndPoints();

protected:
    cv::Mat mPluckerCoord = cv::Mat::zeros(6, 1, CV_32F);
    cv::Mat mOrthonormal = cv::Mat::zeros(4, 1, CV_32F);
    cv::Mat mEndPoints = cv::Mat::zeros(3, 2, CV_32F);

    std::mutex mMutexMapLine;

    KeyFrame* mpFirstObsKF;
    size_t mIndexinFirstKF;
};

#endif // MAPLINE_H